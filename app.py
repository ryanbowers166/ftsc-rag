from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import os
import logging
from typing import List, Optional
from ragie import Ragie
import re
from dotenv import load_dotenv
import vertexai
from vertexai.generative_models import GenerativeModel, ChatSession, GenerationConfig
import csv
import hashlib
from datetime import datetime
import threading
import time

from googleapiclient.discovery import build
from google.auth import default
from google.oauth2 import service_account

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERNAL_API_KEY = os.getenv('INTERNAL_API_KEY')



app = Flask(__name__)
# Configure CORS with restricted origin - UPDATE THE ORIGIN URL BELOW FOR YOUR DOMAIN
CORS(app,
     origins=["http://127.0.0.1:8080"],  # TODO: Replace with your actual domain
     supports_credentials=True,
     allow_headers=["Content-Type", "Authorization", "X-API-Key"],
     methods=["GET", "POST", "OPTIONS"])

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour"],
    storage_uri="memory://"
)

@app.before_request
def check_api_key():
    """Check API key on all requests except health check and OPTIONS"""
    if request.endpoint != 'health' and request.method != 'OPTIONS':
        provided_key = request.headers.get('X-API-Key')

        # Debug logging
        logger.info(f"Endpoint: {request.endpoint}, Method: {request.method}")
        logger.info(f"Expected API key: {INTERNAL_API_KEY[:10] if INTERNAL_API_KEY else 'None'}...")
        logger.info(f"Provided API key: {provided_key[:10] if provided_key else 'None'}...")

        if not provided_key or provided_key != INTERNAL_API_KEY:
            logger.warning(f"Unauthorized request to {request.endpoint} from {request.remote_addr}")
            return jsonify({'success': False, 'error': 'Unauthorized'}), 401

class GoogleDriveHelper:
    def __init__(self, folder_id=None):
        # Get folder ID from environment variable or use default
        self.folder_id = folder_id or os.getenv('GOOGLE_DRIVE_FOLDER_ID', '1AvP6PJzcICFhX2XRKTVJcrtc2lgWvUdN')
        self.service = None
        self.file_cache = {}

    def authenticate(self):
        """Authenticate with Google Drive API"""
        try:
            # Try to use credentials from service account file first
            credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            credentials = None

            if credentials_path:
                try:
                    logger.info(f"Loading Google Drive credentials from: {credentials_path}")
                    credentials = service_account.Credentials.from_service_account_file(
                        credentials_path,
                        scopes=['https://www.googleapis.com/auth/drive.readonly']
                    )
                    logger.info("Service account credentials loaded for Drive API")
                except Exception as e:
                    logger.warning(f"Failed to load service account credentials: {str(e)}")
                    logger.info("Falling back to Application Default Credentials")

            # Fall back to Application Default Credentials if no file path or loading failed
            if not credentials:
                logger.info("Using Application Default Credentials for Drive API")
                credentials, project = default()

            self.service = build('drive', 'v3', credentials=credentials)
            return True
        except Exception as e:
            logger.error(f"Failed to authenticate with Google Drive: {str(e)}")
            return False

    def get_folder_files(self):
        """Get all files from the specified folder"""
        if not self.service:
            if not self.authenticate():
                return {}

        try:
            # Query for files in the specific folder
            query = f"'{self.folder_id}' in parents and trashed=false"

            results = self.service.files().list(
                q=query,
                fields="files(id, name, mimeType, webViewLink, webContentLink)"
            ).execute()

            files = results.get('files', [])

            # Create a mapping of filename to file info
            file_mapping = {}
            for file in files:
                # Store both the exact filename and a normalized version for matching
                filename = file['name']
                file_mapping[filename] = {
                    'id': file['id'],
                    'name': filename,
                    'download_link': file.get('webContentLink', ''),
                    'view_link': file.get('webViewLink', ''),
                    'mime_type': file.get('mimeType', '')
                }

                # Also store without extension for partial matching
                name_without_ext = filename.rsplit('.', 1)[0] if '.' in filename else filename
                if name_without_ext not in file_mapping:
                    file_mapping[name_without_ext] = file_mapping[filename]

            self.file_cache = file_mapping
            logger.info(f"Cached {len(file_mapping)} files from Google Drive")
            return file_mapping

        except Exception as e:
            logger.error(f"Error getting folder files: {str(e)}")
            return {}

    def find_file_links(self, text):
        """Find PDF filenames in text and return their download links"""
        if not self.file_cache:
            self.get_folder_files()

        # Patterns to match various ways PDFs might be referenced
        patterns = [
            r'Source:\s*([^.\n]+\.pdf)',  # "Source: filename.pdf"
            r'([A-Za-z0-9_\-\s]+\.pdf)',   # Any text ending in .pdf
            r'"([^"]+\.pdf)"',             # Quoted PDF filenames
            r'\[([^\]]+\.pdf)\]',          # Bracketed PDF filenames
        ]

        found_files = []

        for pattern in patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                filename = match.group(1).strip()

                # Try exact match first
                if filename in self.file_cache:
                    found_files.append(self.file_cache[filename])
                    continue

                # Try partial matching (without extension, case insensitive)
                filename_lower = filename.lower()
                for cached_name, file_info in self.file_cache.items():
                    if (cached_name.lower() == filename_lower or
                        cached_name.lower().startswith(filename_lower.rsplit('.', 1)[0])):
                        found_files.append(file_info)
                        break

        # Remove duplicates
        unique_files = []
        seen_ids = set()
        for file_info in found_files:
            if file_info['id'] not in seen_ids:
                unique_files.append(file_info)
                seen_ids.add(file_info['id'])

        return unique_files

# Query logging infrastructure
_csv_lock = threading.Lock()
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
_log_file = os.path.join(_log_dir, 'query_metadata.csv')

def log_query_metadata(
    query_text: str,
    query_length_chars: int,
    query_word_count: int,
    is_follow_up: bool,
    conversation_history_length: int,
    response_text: str,
    response_length_chars: int,
    response_length_words: int,
    response_length_tokens_est: int,
    num_chunks_retrieved: int,
    num_sources_found: int,
    source_documents: List[str],
    processing_time_ms: float,
    had_results: bool,
    status: str,
    error_message: str = ""
):
    """
    Thread-safe logging of query metadata to CSV file.
    Only logs if ENABLE_QUERY_LOGGING environment variable is set to 'true'.
    """
    # Check if logging is enabled
    # if os.getenv('ENABLE_QUERY_LOGGING', 'true').lower() != 'true':
    #     logger.info(f"ENABLE_QUERY_LOGGING false, not logging metadata")
    #     return

    try:
        # Create logs directory if it doesn't exist
        os.makedirs(_log_dir, exist_ok=True)

        # Hash the query for privacy (SHA256, first 16 chars)
        query_hash = hashlib.sha256(query_text.encode('utf-8')).hexdigest()[:16]

        # Format source documents as pipe-separated string
        source_docs_str = '|'.join(source_documents) if source_documents else ''

        # Prepare the row data
        row_data = {
            'timestamp': datetime.utcnow().isoformat(),
            #'query_hash': query_hash,
            'query_length_chars': query_length_chars,
            'query_word_count': query_word_count,
            'is_follow_up': is_follow_up,
            'conversation_history_length': conversation_history_length,
            'response_length_chars': response_length_chars,
            'response_length_words': response_length_words,
            'response_length_tokens_est': response_length_tokens_est,
            'num_chunks_retrieved': num_chunks_retrieved,
            'num_sources_found': num_sources_found,
            #'source_documents': source_docs_str,
            'processing_time_ms': round(processing_time_ms, 2),
            'had_results': had_results,
            'status': status,
            'error_message': error_message
        }

        # Thread-safe CSV writing
        with _csv_lock:
            # Check if file exists to determine if we need to write headers
            file_exists = os.path.isfile(_log_file)

            with open(_log_file, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = [
                    'timestamp', 'query_hash', 'query_length_chars', 'query_word_count',
                    'is_follow_up', 'conversation_history_length', 'response_length_chars',
                    'response_length_words', 'response_length_tokens_est', 'num_chunks_retrieved',
                    'num_sources_found', 'source_documents', 'processing_time_ms',
                    'had_results', 'status', 'error_message'
                ]

                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()

                # Write the data row
                writer.writerow(row_data)

        logger.debug(f"Query metadata logged successfully: {query_hash}")

    except Exception as e:
        # Log errors but don't fail the request
        logger.error(f"Error logging query metadata: {str(e)}")

class RAGSystem:
    def __init__(self):
        self.ragie_client = None
        self.gemini_model = None
        self.initialized = False
        self.drive_helper = GoogleDriveHelper()
        self.ragie_api_key = os.getenv('RAGIE_API_KEY')
        self.gcp_project = os.getenv('GOOGLE_CLOUD_PROJECT')
        self.gcp_region = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')
        self.gcp_credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.credentials = None
        self.connection_name = "FTSC LLM Data Processed"  # Google Drive connection name in Ragie

        self.temperature = 0.5
        self.llm_model = "gemini-2.5-pro"
        self.max_response_length = 5000


    def initialize(self):
        """Initialize the RAG system with Ragie and Vertex AI Gemini"""
        try:
            if self.initialized:
                logger.info("RAG system already initialized")
                return True

            # Initialize Ragie
            if not self.ragie_api_key:
                logger.error("RAGIE_API_KEY environment variable not set")
                return False

            logger.info("Initializing Ragie client...")
            self.ragie_client = Ragie(auth=self.ragie_api_key)
            logger.info("Ragie client initialized successfully!")

            # Load credentials if path is provided
            if self.gcp_credentials_path:
                try:
                    logger.info(f"Loading credentials from: {self.gcp_credentials_path}")
                    self.credentials = service_account.Credentials.from_service_account_file(
                        self.gcp_credentials_path,
                        scopes=['https://www.googleapis.com/auth/cloud-platform']
                    )
                    logger.info("Service account credentials loaded successfully!")
                except Exception as e:
                    logger.error(f"Failed to load credentials from {self.gcp_credentials_path}: {str(e)}")
                    logger.info("Falling back to Application Default Credentials")
                    self.credentials = None
            else:
                logger.info("No GOOGLE_APPLICATION_CREDENTIALS path provided, using Application Default Credentials")
                self.credentials = None

            # Initialize Vertex AI
            if not self.gcp_project:
                logger.error("GOOGLE_CLOUD_PROJECT environment variable not set")
                return False

            logger.info(f"Initializing Vertex AI with project: {self.gcp_project}, region: {self.gcp_region}")
            vertexai.init(
                project=self.gcp_project,
                location=self.gcp_region,
                credentials=self.credentials
            )

            # Initialize Gemini model
            generation_config = GenerationConfig(temperature=self.temperature)
            self.gemini_model = GenerativeModel(
                model_name=self.llm_model,
                #tools=
                generation_config=generation_config
            )

            logger.info(f"{self.llm_model} model initialized successfully!")

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            return False

    def health_check(self) -> bool:
        """Check if the RAG system is healthy and re-initialize if needed"""
        try:
            if not self.initialized or not self.ragie_client or not self.gemini_model:
                logger.warning("RAG system not healthy, attempting re-initialization...")
                return self.initialize()

            logger.info("RAG system health check passed")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            logger.warning("Attempting to re-initialize RAG system...")
            return self.initialize()

    def retrieve_chunks(self, query: str):
        """Retrieve chunks from Ragie using simplified structure"""
        retrieval_res = self.ragie_client.retrievals.retrieve(request={
            "query": query,
            "rerank": True,
            "top_k": 10,
            "max_chunks_per_document": 0
        })

        if retrieval_res:
            return retrieval_res.scored_chunks
        return []

    def format_conversation_history(self, conversation_history: List[dict]) -> str:
        """Format conversation history for inclusion in the prompt"""
        if not conversation_history:
            return ""

        formatted_history = "\n\nPREVIOUS CONVERSATION CONTEXT:\n"
        for i, exchange in enumerate(conversation_history, 1):
            formatted_history += f"\n--- Previous Exchange {i} ---\n"
            formatted_history += f"Human: {exchange['query']}\n"
            formatted_history += f"Assistant: {exchange['response']}\n"

        formatted_history += "\n--- End of Previous Context ---\n"
        formatted_history += "\nPlease use this conversation context to provide a coherent response to the new query below. If the new query refers to previous topics or asks for clarification, use the context above to maintain continuity.\n\n"

        return formatted_history

    def clean_hallucinated_sources(self, text):
        """Remove hallucinated generic source citations that don't reference actual files"""

        # Pattern to match "Source: [text]" and capture the content after "Source:"
        # We'll then check if that content contains file extensions
        def source_replacer(match):
            source_content = match.group(1).strip()

            # Check if the source content contains a file extension
            has_file_extension = re.search(r'\.(?:pdf|doc|docx|txt|xlsx|xls|ppt|pptx)\b', source_content, re.IGNORECASE)

            if has_file_extension:
                # Keep this source - it's a real file reference
                return match.group(0)
            else:
                # Remove this source - it's hallucinated
                return ''

        # Pattern to match "Source: " followed by everything until newline or end of string
        pattern = r'\s*Source:\s*([^\n]*?)(?=\n|$)'

        # Apply the replacement function
        cleaned_text = re.sub(pattern, source_replacer, text, flags=re.IGNORECASE)

        # Clean up any extra spaces on the same line, but preserve newlines
        cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)

        # Clean up any double newlines
        cleaned_text = re.sub(r'\n\s*\n+', '\n\n', cleaned_text)
        cleaned_text = cleaned_text.strip()

        return cleaned_text

    def query_with_sources(self, user_query: str, conversation_history: List[dict] = None) -> dict:
        """Process a query using the RAG system with conversation history support"""
        # Health check before processing query
        if not self.health_check():
            raise Exception("RAG system health check failed and could not be re-initialized")

        try:
            logger.info(f"Processing query with {len(conversation_history) if conversation_history else 0} previous exchanges: {user_query[:100]}...")

            # Step 1: Retrieve relevant chunks from Ragie
            logger.info("Retrieving relevant document chunks from Ragie...")
            scored_chunks = self.retrieve_chunks(user_query)

            # Extract chunks and format them
            chunks_text = ""
            num_chunks_retrieved = len(scored_chunks) if scored_chunks else 0

            if scored_chunks:
                logger.info(f"Retrieved {len(scored_chunks)} chunks")
                for i, chunk in enumerate(scored_chunks, 1):
                    chunk_content = chunk.text if hasattr(chunk, 'text') else str(chunk)
                    # Include document name if available
                    doc_name = ""
                    if hasattr(chunk, 'document') and chunk.document:
                        if hasattr(chunk.document, 'name'):
                            doc_name = f" [Source: {chunk.document.name}]"
                        elif hasattr(chunk.document, 'metadata') and isinstance(chunk.document.metadata, dict):
                            doc_name = f" [Source: {chunk.document.metadata.get('name', 'Unknown')}]"

                    chunks_text += f"\n--- Chunk {i}{doc_name} ---\n{chunk_content}\n"
            else:
                logger.warning("No chunks retrieved from Ragie")
                chunks_text = "No relevant documents found."

            # Step 2: Generate response using Gemini
            logger.info(f"Generating response with {self.llm_model}...")

            if scored_chunks and len(scored_chunks) > 0:
                # Build conversation context for Gemini
                # Disable response validation to allow responses that might be flagged as recitation
                chat = self.gemini_model.start_chat(response_validation=False)

                # Build the full prompt with the new DOCUMENT/QUESTION/INSTRUCTIONS format
                full_prompt = "DOCUMENT:\n"
                full_prompt += chunks_text + "\n\n"

                # Add conversation history if provided
                if conversation_history:
                    full_prompt += "PREVIOUS CONVERSATION:\n"
                    for i, exchange in enumerate(conversation_history, 1):
                        full_prompt += f"User ({i}): {exchange.get('query', '')}\n"
                        full_prompt += f"Assistant ({i}): {exchange.get('response', '')}\n"
                    full_prompt += "\n"

                full_prompt += "QUESTION:\n"
                full_prompt += user_query + "\n\n"

                full_prompt += """INSTRUCTIONS:
Answer the user's QUESTION using the DOCUMENT text above.
Keep your answer grounded in the facts of the DOCUMENT.
If the DOCUMENT doesn't contain the facts to answer the QUESTION return "Sorry, I couldn't find any sources in the database relevant to your question."

Additional guidelines:
- You are helping a flight test professional analyze technical papers and documentation about flight test techniques, procedures, considerations, and lessons learned.
- ALWAYS cite the exact source document name for every piece of information. Format sources as: "Source: [exact filename from retrieval]"
- If you pull multiple pieces of information from the same source, you can cite the source once.
- Never create or invent source names - only use what's provided in the DOCUMENT above.
- For broad topics (e.g., "Autonomous vehicles"), provide a comprehensive overview covering multiple aspects.
- For specific queries (e.g., "high altitude flight test sources"), focus on the specific topic requested.
- CONVERSATION CONTINUITY: If the QUESTION references previous exchanges or asks for clarification about earlier topics (shown in PREVIOUS CONVERSATION above), use that context to maintain coherent discussion flow.
- If the user asks follow-up questions like "tell me more about that" or "what about safety considerations", refer to the previous context to understand what "that" refers to.
"""

                # Call Gemini API
                try:
                    response = chat.send_message(
                        full_prompt,
                        generation_config={
                            "temperature": self.temperature,
                            "max_output_tokens": self.max_response_length,
                        }
                    )

                    response_text = response.text

                except Exception as e:
                    logger.error(f"Gemini API error: {str(e)}")
                    response_text = f"Error generating response: {str(e)}\n\nRetrieved chunks:\n{chunks_text}"

            else:
                response_text = "Sorry, I couldn't find any sources in the database relevant to your question."

            # Find source files in the response
            source_files = self.drive_helper.find_file_links(response_text)

            # Clean hallucinated sources
            response_text = self.clean_hallucinated_sources(response_text)

            logger.info(f"Query processed successfully with {len(source_files)} source files found")

            return {
                'response': response_text,
                'sources': source_files,
                'num_chunks_retrieved': num_chunks_retrieved  # Add for logging
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def direct_retrieval_query(self, query_text: str, top_k: int = 6):
        """Perform direct context retrieval using Ragie"""
        if not self.ragie_client:
            raise Exception("Ragie client not initialized")

        try:
            logger.info(f"Performing direct retrieval for: {query_text[:100]}...")

            # Use Ragie's retrieval endpoint with simplified structure
            retrieval_res = self.ragie_client.retrievals.retrieve(
                request={
                    "query": query_text,
                    "rerank": True,
                    "top_k": top_k,
                    "max_chunks_per_document": 0
                }
            )

            logger.info("Direct retrieval completed successfully")
            if retrieval_res:
                return retrieval_res.scored_chunks
            return []

        except Exception as e:
            logger.error(f"Error in direct retrieval: {str(e)}")
            raise

# Initialize the RAG system
rag_system = RAGSystem()


def initialize_rag_system():
    """Initialize the RAG system with Ragie"""
    global rag_system

    try:
        # Initialize Ragie client
        success = rag_system.initialize()

        if success:
            logger.info("RAG system initialized successfully!")
            return True
        else:
            logger.error("Failed to initialize RAG system")
            return False

    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False


# HTML template loading function
def load_template():
    """Load the HTML template"""
    try:
        with open('_archive/template.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error("template.html not found")
        return """<!DOCTYPE html>
        <html><head><meta charset="UTF-8"><title>Error</title></head>
        <body><h1>Template file not found</h1>
        <p>Create template.html file in your project directory</p></body></html>"""

@app.route('/')
def index():
    """Serve the main page"""
    return load_template()


@app.route('/status')
def status():
    """Check if the RAG system is initialized with health check"""
    # Perform health check
    is_healthy = rag_system.health_check()

    return jsonify({
        'initialized': rag_system.initialized,
        'healthy': is_healthy,
        'has_client': rag_system.ragie_client is not None,
        'connection_name': rag_system.connection_name
    })


@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the RAG system"""
    try:
        success = initialize_rag_system()
        if success:
            return jsonify({
                'success': True,
                'message': 'RAG system initialized successfully!',
                'connection_name': rag_system.connection_name
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to initialize RAG system'
            })
    except Exception as e:
        logger.error(f"Error in initialize endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error initializing RAG system: {str(e)}'
        }), 500


# Query endpoint with conversation history support
@app.route('/query', methods=['POST'])
@limiter.limit("20 per minute")  # Rate limit: 20 queries per minute per IP
def query():
    """Process a research query with conversation history support"""
    global rag_system

    logger.info(f"Query endpoint called. Initialized: {rag_system.initialized}")

    # Start timing
    start_time = time.time()

    # Initialize metadata variables
    user_query = ""
    conversation_history = []
    response_text = ""
    source_files = []
    num_chunks_retrieved = 0
    status = "error"
    error_message = ""

    try:
        data = request.get_json()
        logger.info(f"Received data keys: {list(data.keys()) if data else 'No data'}")

        if not data:
            logger.error("No JSON data received")
            error_message = "No data received"
            return jsonify({'error': error_message}), 400

        user_query = data.get('query', '').strip()
        conversation_history = data.get('conversation_history', [])

        # Input validation
        MAX_QUERY_LENGTH = 2000
        if len(user_query) > MAX_QUERY_LENGTH:
            error_message = f'Query too long (max {MAX_QUERY_LENGTH} characters)'
            return jsonify({'error': error_message}), 400

        logger.info(f"User query (truncated): {user_query[:100]}...")
        logger.info(f"Conversation history length: {len(conversation_history)}")

        if not user_query:
            logger.error("Empty query received")
            error_message = "Query cannot be empty"
            return jsonify({'error': error_message}), 400

        # Process the query using RAG system with conversation history
        result = rag_system.query_with_sources(user_query, conversation_history)

        # Extract metadata from result
        response_text = result.get('response', '')
        source_files = result.get('sources', [])
        num_chunks_retrieved = result.get('num_chunks_retrieved', 0)
        status = "success"

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Calculate metrics
        query_length_chars = len(user_query)
        query_word_count = len(user_query.split())
        is_follow_up = len(conversation_history) > 0
        conversation_history_length = len(conversation_history)
        response_length_chars = len(response_text)
        response_length_words = len(response_text.split())
        response_length_tokens_est = response_length_chars // 4  # Rough estimate
        num_sources_found = len(source_files)
        source_documents = [sf.get('name', '') for sf in source_files]
        had_results = num_chunks_retrieved > 0

        # Log query metadata
        log_query_metadata(
            query_text=user_query,
            query_length_chars=query_length_chars,
            query_word_count=query_word_count,
            is_follow_up=is_follow_up,
            conversation_history_length=conversation_history_length,
            response_text=response_text,
            response_length_chars=response_length_chars,
            response_length_words=response_length_words,
            response_length_tokens_est=response_length_tokens_est,
            num_chunks_retrieved=num_chunks_retrieved,
            num_sources_found=num_sources_found,
            source_documents=source_documents,
            processing_time_ms=processing_time_ms,
            had_results=had_results,
            status=status,
            error_message=""
        )

        logger.info(f"Returning result with {len(result['response'])} characters and {len(result['sources'])} sources")

        # Remove num_chunks_retrieved from response (only used for logging)
        result.pop('num_chunks_retrieved', None)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        error_message = str(e)
        status = "error"

        # Calculate processing time even for errors
        processing_time_ms = (time.time() - start_time) * 1000

        # Log error metadata if we have a query
        if user_query:
            query_length_chars = len(user_query)
            query_word_count = len(user_query.split())
            is_follow_up = len(conversation_history) > 0
            conversation_history_length = len(conversation_history)

            log_query_metadata(
                query_text=user_query,
                query_length_chars=query_length_chars,
                query_word_count=query_word_count,
                is_follow_up=is_follow_up,
                conversation_history_length=conversation_history_length,
                response_text="",
                response_length_chars=0,
                response_length_words=0,
                response_length_tokens_est=0,
                num_chunks_retrieved=0,
                num_sources_found=0,
                source_documents=[],
                processing_time_ms=processing_time_ms,
                had_results=False,
                status=status,
                error_message=error_message
            )

        return jsonify({'error': f'Error processing query: {str(e)}'}), 500


@app.route('/direct-retrieval', methods=['POST'])
def direct_retrieval():
    """Perform direct context retrieval without generation with health check"""
    global rag_system

    # Check health before processing
    if not rag_system.health_check():
        return jsonify({'error': 'RAG system is not healthy and could not be recovered'}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        query_text = data.get('query', '').strip()
        top_k = data.get('top_k', 6)

        if not query_text:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Perform direct retrieval
        scored_chunks = rag_system.direct_retrieval_query(
            query_text,
            top_k=top_k
        )

        # Format chunks for response
        chunks_info = []
        for chunk in scored_chunks:
            chunk_data = {
                'text': chunk.text if hasattr(chunk, 'text') else str(chunk),
                'score': chunk.score if hasattr(chunk, 'score') else None
            }
            if hasattr(chunk, 'document') and chunk.document:
                if hasattr(chunk.document, 'name'):
                    chunk_data['document_name'] = chunk.document.name
            chunks_info.append(chunk_data)

        return jsonify({
            'chunks': chunks_info,
            'count': len(scored_chunks)
        })

    except Exception as e:
        logger.error(f"Error in direct retrieval: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error in direct retrieval: {str(e)}'}), 500


@app.route('/health')
def health():
    """Enhanced health check endpoint for Cloud Run - always returns 200 for liveness probe"""
    is_healthy = rag_system.health_check()

    # Always return 200 for Cloud Run liveness probes
    # Cloud Run needs a 200 status to keep the container running
    return jsonify({
        'status': 'healthy' if is_healthy else 'initializing',
        'initialized': rag_system.initialized,
        'has_client': rag_system.ragie_client is not None,
        'connection_name': rag_system.connection_name,
        'auto_recovery': is_healthy
    }), 200


@app.route('/test')
def test():
    """Simple test endpoint with health check"""
    is_healthy = rag_system.health_check()

    return jsonify({
        'message': 'Flask server is working!',
        'initialized': rag_system.initialized,
        'healthy': is_healthy,
        'has_client': rag_system.ragie_client is not None,
        'connection_name': rag_system.connection_name
    })


@app.route('/refresh-files', methods=['POST'])
def refresh_files():
    """Refresh the Google Drive file cache"""
    try:
        files = rag_system.drive_helper.get_folder_files()
        return jsonify({
            'success': True,
            'message': f'Refreshed cache with {len(files)} files',
            'file_count': len(files)
        })
    except Exception as e:
        logger.error(f"Error refreshing files: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add endpoint to test file detection
@app.route('/test-file-detection', methods=['POST'])
def test_file_detection():
    """Test file detection in a sample text"""
    try:
        data = request.get_json()
        test_text = data.get('text', '')

        if not test_text:
            return jsonify({'error': 'No text provided'}), 400

        found_files = rag_system.drive_helper.find_file_links(test_text)

        return jsonify({
            'found_files': found_files,
            'count': len(found_files)
        })

    except Exception as e:
        logger.error(f"Error in file detection test: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Conversation management endpoints
@app.route('/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history (this is mainly handled client-side, but included for completeness)"""
    try:
        return jsonify({
            'success': True,
            'message': 'Conversation cleared successfully'
        })
    except Exception as e:
        logger.error(f"Error clearing conversation: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Initialize RAG system on startup
    logger.info("Starting Flask app and initializing RAG system...")

    # Try to initialize, but don't fail if it doesn't work initially
    try:
        drive_helper = GoogleDriveHelper()
        initialize_rag_system()
    except Exception as e:
        logger.error(f"Initial RAG system initialization failed: {str(e)}")
        logger.info("RAG system will be initialized on first request")

    # Run the app
    port = int(os.environ.get('PORT', 5001))

    app.run(host='0.0.0.0', port=port, debug=False)
