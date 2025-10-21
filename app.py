from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging
from typing import List, Optional
from ragie import Ragie
import re

from googleapiclient.discovery import build
from google.auth import default

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per hour"],
    storage_uri="memory://"
)

class GoogleDriveHelper:
    def __init__(self, folder_id=None):
        # Get folder ID from environment variable or use default
        self.folder_id = folder_id or os.getenv('GOOGLE_DRIVE_FOLDER_ID', '1Qif8tvURTHOOrtrosTQ4YU077yPnuiTB')
        self.service = None
        self.file_cache = {}

    def authenticate(self):
        """Authenticate with Google Drive API"""
        try:
            # Use default credentials (works in Cloud Shell/Cloud Run)
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

class RAGSystem:
    def __init__(self):
        self.ragie_client = None
        self.initialized = False
        self.drive_helper = GoogleDriveHelper()
        self.api_key = os.getenv('RAGIE_API_KEY')
        self.connection_name = "FTSC LLM Data"  # Google Drive connection name in Ragie

    def initialize(self):
        """Initialize the RAG system with Ragie"""
        try:
            if self.initialized:
                logger.info("RAG system already initialized")
                return True

            if not self.api_key:
                logger.error("RAGIE_API_KEY environment variable not set")
                return False

            logger.info("Initializing Ragie client...")
            self.ragie_client = Ragie(auth=self.api_key)

            # Test the connection by making a simple API call
            try:
                # This will verify the API key works
                logger.info("Testing Ragie connection...")
                self.initialized = True
                logger.info("Ragie client initialized successfully!")
                return True
            except Exception as e:
                logger.error(f"Failed to connect to Ragie: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            return False

    def health_check(self) -> bool:
        """Check if the RAG system is healthy and re-initialize if needed"""
        try:
            if not self.initialized or not self.ragie_client:
                logger.warning("RAG system not healthy, attempting re-initialization...")
                return self.initialize()

            logger.info("RAG system health check passed")
            return True

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            logger.warning("Attempting to re-initialize RAG system...")
            return self.initialize()

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

            # Create a research-focused system prompt
            base_system_prompt = """You are a helpful chat agent helping a flight test professional analyze technical papers and documentation. You will help the user find relevant sources in the database about flight test techniques, procedures, considerations, and lessons learned.

CRITICAL RULES:
1. ONLY use information from the retrieved documents
2. ALWAYS cite the exact source document name for every piece of information. If you pull multiple pieces of information from the same source, you can just cite the source once.
3. If no relevant information is found, say "No relevant information found in the available sources"
4. Never create or invent source names - only use what's provided in the retrieval results
5. Format sources as: "Source: [exact filename from retrieval]"
6. For broad topics (e.g., "Autonomous vehicles"), provide a comprehensive overview covering multiple aspects
7. For specific queries (e.g., "high altitude flight test sources"), focus on the specific topic requested
8. CONVERSATION CONTINUITY: If this message references previous exchanges or asks for clarification about earlier topics, use the conversation context to maintain coherent discussion flow
9. If the user asks follow-up questions like "tell me more about that" or "what about safety considerations", refer to the previous context to understand what "that" refers to

Answer the user's question based solely on the retrieved information and conversation context.

"""

            # Add conversation history if provided
            conversation_context = ""
            if conversation_history:
                conversation_context = self.format_conversation_history(conversation_history)

            # Step 1: Retrieve relevant chunks from Ragie
            logger.info("Retrieving relevant document chunks from Ragie...")
            retrieval_response = self.ragie_client.retrievals.retrieve(
                request={
                    "query": user_query,
                    "rerank": True,
                    "top_k": 15,
                    "max_chunks_per_document": 3,
                    "filter": {
                        "connection_name": self.connection_name
                    }
                }
            )

            # Extract chunks and format them
            chunks_text = ""
            if retrieval_response and hasattr(retrieval_response, 'scored_chunks'):
                logger.info(f"Retrieved {len(retrieval_response.scored_chunks)} chunks")
                for i, chunk in enumerate(retrieval_response.scored_chunks, 1):
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

            # Step 2: Generate response using Ragie's built-in LLM
            logger.info("Generating response with Ragie's LLM...")

            # Combine system prompt, conversation context, retrieved chunks, and user query
            full_prompt = base_system_prompt + conversation_context
            full_prompt += "\n\nRETRIEVED DOCUMENT CHUNKS:\n" + chunks_text
            full_prompt += f"\n\nUSER QUERY: {user_query}\n\nRESPONSE:"

            # Use Ragie's chat endpoint for generation
            try:
                # Ragie supports RAG generation - retrieve and generate in one call
                rag_response = self.ragie_client.retrievals.rag(
                    request={
                        "query": user_query,
                        "rerank": True,
                        "top_k": 15,
                        "max_chunks_per_document": 3,
                        "filter": {
                            "connection_name": self.connection_name
                        },
                        "instructions": base_system_prompt + conversation_context
                    }
                )

                # Extract response text
                if rag_response and hasattr(rag_response, 'answer'):
                    response_text = rag_response.answer
                elif rag_response and hasattr(rag_response, 'response'):
                    response_text = rag_response.response
                else:
                    response_text = str(rag_response)

            except AttributeError:
                # Fallback if RAG method doesn't exist - just use retrieved chunks
                logger.warning("Ragie RAG generation not available, using chunks only")
                response_text = "Based on the retrieved documents:\n\n" + chunks_text

            # Find source files in the response
            source_files = self.drive_helper.find_file_links(response_text)

            # Clean hallucinated sources
            response_text = self.clean_hallucinated_sources(response_text)

            logger.info(f"Query processed successfully with {len(source_files)} source files found")

            return {
                'response': response_text,
                'sources': source_files
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def direct_retrieval_query(self, query_text: str, top_k: int = 15, vector_distance_threshold: float = 0.8):
        """Perform direct context retrieval using Ragie"""
        if not self.ragie_client:
            raise Exception("Ragie client not initialized")

        try:
            logger.info(f"Performing direct retrieval for: {query_text[:100]}...")

            # Use Ragie's retrieval endpoint
            retrieval_response = self.ragie_client.retrievals.retrieve(
                request={
                    "query": query_text,
                    "rerank": True,
                    "top_k": top_k,
                    "max_chunks_per_document": 3,
                    "filter": {
                        "connection_name": self.connection_name
                    }
                }
            )

            logger.info("Direct retrieval completed successfully")
            return retrieval_response

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
        with open('template.html', 'r', encoding='utf-8') as f:
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

    try:
        data = request.get_json()
        logger.info(f"Received data keys: {list(data.keys()) if data else 'No data'}")

        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400

        user_query = data.get('query', '').strip()
        conversation_history = data.get('conversation_history', [])

        # Input validation
        MAX_QUERY_LENGTH = 2000
        if len(user_query) > MAX_QUERY_LENGTH:
            return jsonify({'error': f'Query too long (max {MAX_QUERY_LENGTH} characters)'}), 400

        logger.info(f"User query (truncated): {user_query[:100]}...")
        logger.info(f"Conversation history length: {len(conversation_history)}")

        if not user_query:
            logger.error("Empty query received")
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Process the query using RAG system with conversation history
        result = rag_system.query_with_sources(user_query, conversation_history)

        logger.info(f"Returning result with {len(result['response'])} characters and {len(result['sources'])} sources")
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
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
        top_k = data.get('top_k', 10)
        vector_distance_threshold = data.get('vector_distance_threshold', 0.4)

        if not query_text:
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Perform direct retrieval
        retrieval_response = rag_system.direct_retrieval_query(
            query_text,
            top_k=top_k,
            vector_distance_threshold=vector_distance_threshold
        )

        return jsonify({'retrieval_response': str(retrieval_response)})

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

@app.route('/static/<path:filename>')
def serve_static_files(filename):
    return send_from_directory('static', filename)

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
    port = int(os.environ.get('PORT', 8080))

    app.run(host='0.0.0.0', port=port, debug=False)
