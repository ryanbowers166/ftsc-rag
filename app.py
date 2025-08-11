from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import os
import logging
from typing import List, Tuple, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool
from vertexai import rag
from google.cloud import aiplatform
import json
import tempfile

from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.auth import default
import re
from urllib.parse import quote

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

class GoogleDriveHelper:
    def __init__(self, folder_id="1Qif8tvURTHOOrtrosTQ4YU077yPnuiTB"):
        self.folder_id = folder_id
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
        self.rag_model = None
        self.rag_corpus = None
        self.rag_retrieval_tool = None
        self.initialized = False
        self.PROJECT_ID = "ftscrag"
        self.LOCATION = "us-central1"
        self.CORPUS_DISPLAY_NAME = "FTSC Research Papers Corpus"  # Consistent corpus name
        self.drive_helper = GoogleDriveHelper()

    def setup_authentication(self):
        """Setup Google Cloud authentication"""
        try:
            # In Cloud Shell, use Application Default Credentials
            logger.info("Using Application Default Credentials in Cloud Shell")

            # Check if we can access the project
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT') or os.getenv('DEVSHELL_PROJECT_ID')
            if project_id:
                logger.info(f"Found project ID: {project_id}")
                self.PROJECT_ID = project_id  # Use the environment project ID
                return True

            # If no project ID found, still try to proceed with default
            logger.info("No project ID environment variable found, using default project ID")
            return True

        except Exception as e:
            logger.error(f"Authentication setup failed: {str(e)}")
            return False

    def find_existing_corpus(self) -> Optional[str]:
        """Find existing corpus by display name"""
        try:
            logger.info(f"Searching for existing corpus: {self.CORPUS_DISPLAY_NAME}")

            # List all corpora and find one with matching display name
            corpora = rag.list_corpora()

            for corpus in corpora:
                if corpus.display_name == self.CORPUS_DISPLAY_NAME:
                    logger.info(f"Found existing corpus: {corpus.name}")
                    return corpus.name

            logger.info("No existing corpus found")
            return None

        except Exception as e:
            logger.error(f"Error searching for existing corpus: {str(e)}")
            return None

    def initialize(self):
        """Initialize the RAG system with Vertex AI"""
        try:
            if self.initialized:
                logger.info("RAG system already initialized")
                return True

            if not self.setup_authentication():
                logger.error("Authentication setup failed")
                return False

            logger.info("Initializing Vertex AI...")
            # Initialize Vertex AI once per session
            vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)
            aiplatform.init(project=self.PROJECT_ID, location=self.LOCATION)

            # First, try to find existing corpus
            existing_corpus_name = self.find_existing_corpus()

            if existing_corpus_name:
                logger.info("Using existing corpus")
                success = self.load_existing_corpus(existing_corpus_name)
            else:
                logger.info("Creating new corpus")
                success = self.create_corpus_from_drive()

            if success:
                # Setup the model after loading/creating corpus
                model_success = self.setup_model()
                if model_success:
                    self.initialized = True
                    logger.info("RAG system initialized successfully!")
                    return True
                else:
                    logger.error("Failed to setup model after corpus creation/loading")
                    return False
            else:
                logger.error("Failed to create or load corpus")
                return False

        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            return False

    def create_corpus_from_drive(self) -> bool:
        """Create RAG corpus from Google Drive folder using new syntax"""
        try:
            logger.info("Creating corpus from Google Drive folder...")

            # Google Drive folder URL
            #drive_folder_url = "https://drive.google.com/drive/folders/1Qif8tvURTHOOrtrosTQ4YU077yPnuiTB"
            drive_folder_url = "https://drive.google.com/drive/u/2/folders/1f2UR4a-Anf9aExc3DO9DEsSVX-cNOhgK"

            # Configure embedding model using new syntax
            embedding_model_config = rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            )

            # Create RagCorpus using new syntax with consistent display name
            self.rag_corpus = rag.create_corpus(
                display_name=self.CORPUS_DISPLAY_NAME,  # Use consistent name
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config
                ),
            )

            logger.info(f"Corpus created: {self.rag_corpus.name}")

            # Import files from Google Drive folder using new syntax
            logger.info("Importing files from Google Drive folder...")
            paths = [drive_folder_url]

            rag.import_files(
                self.rag_corpus.name,
                paths,
                # Optional transformation config
                transformation_config=rag.TransformationConfig(
                    chunking_config=rag.ChunkingConfig(
                        chunk_size=512,
                        chunk_overlap=150,
                    ),
                ),
                max_embedding_requests_per_min=1000,  # Optional
            )

            logger.info("Files imported successfully from Google Drive")
            return True

        except Exception as e:
            logger.error(f"Failed to create corpus from Drive: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: Make sure the Google Drive folder is accessible to your service account")
            return False

    def load_existing_corpus(self, corpus_name: str) -> bool:
        """Load an existing RAG corpus"""
        try:
            logger.info(f"Loading existing corpus: {corpus_name}")
            # Get the corpus by name
            self.rag_corpus = rag.get_corpus(name=corpus_name)
            logger.info(f"Loaded corpus: {self.rag_corpus.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load corpus: {str(e)}")
            return False

    def health_check(self) -> bool:
        """Check if the RAG system is healthy and re-initialize if needed"""
        try:
            if not self.initialized or not self.rag_model or not self.rag_corpus:
                logger.warning("RAG system not healthy, attempting re-initialization...")
                return self.initialize()

            # Test if we can still access the corpus
            if self.rag_corpus:
                # Try to access corpus info to verify it's still valid
                corpus_name = self.rag_corpus.name
                logger.info(f"RAG system health check passed for corpus: {corpus_name}")
                return True

            return False

        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            logger.warning("Attempting to re-initialize RAG system...")
            return self.initialize()

    def setup_model(self, top_k: int = 15, vector_distance_threshold: float = 0.8,
                   llm_model_name: str = "gemini-2.0-flash-001", temperature: float = 0.5):
        """Setup the RAG model with retrieval tool using new syntax"""
        try:
            if not self.rag_corpus:
                logger.error("No corpus available. Create or load a corpus first.")
                return False

            logger.info("Setting up RAG model with new syntax...")

            # Direct context retrieval configuration
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=top_k,  # Optional
                filter=rag.Filter(vector_distance_threshold=vector_distance_threshold),  # Optional
            )

            # Create a RAG retrieval tool using new syntax
            self.rag_retrieval_tool = Tool.from_retrieval(
                retrieval=rag.Retrieval(
                    source=rag.VertexRagStore(
                        rag_resources=[
                            rag.RagResource(
                                rag_corpus=self.rag_corpus.name,  # Currently only 1 corpus is allowed
                                # Optional: supply IDs from `rag.list_files()`.
                                # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                            )
                        ],
                        rag_retrieval_config=rag_retrieval_config,
                    ),
                )
            )

            # Create a Gemini model instance
            generation_config = GenerationConfig(temperature=temperature)
            self.rag_model = GenerativeModel(
                model_name=llm_model_name,
                tools=[self.rag_retrieval_tool],
                generation_config=generation_config
            )

            logger.info("RAG model setup completed successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to setup model: {str(e)}")
            return False

    def query_with_sources(self, user_query: str) -> str:
        """Process a query using the RAG system with new syntax"""
        # Health check before processing query
        if not self.health_check():
            raise Exception("RAG system health check failed and could not be re-initialized")

        try:
            logger.info(f"Processing query: {user_query[:100]}...")

            # Create a research-focused system prompt
            system_prompt = """You are a helpful chat agent helping a flight test professional analyze technical papers and documentation. You will help the user find relevant sources in the database about flight test techniques, procedures, considerations, and lessons learned.

CRITICAL RULES:
1. ONLY use information from the retrieved documents
2. ALWAYS cite the exact source document name for every piece of information. If you pull multiple pieces of information from the same source, you can just cite the source once.
3. If no relevant information is found, say "No relevant information found in the available sources"
4. Never create or invent source names - only use what's provided in the retrieval results
5. Format sources as: "Source: [exact filename from retrieval]"
6. For broad topics (e.g., "Autonomous vehicles"), provide a comprehensive overview covering multiple aspects
7. For specific queries (e.g., "high altitude flight test sources"), focus on the specific topic requested

Answer the user's question based solely on the retrieved information.

User query: """

            full_query = system_prompt + user_query

            # Generate response using the RAG model with new syntax
            response = self.rag_model.generate_content(full_query)
            response_text = response.text

            # Find source files in the response
            source_files = self.drive_helper.find_file_links(response_text)

            logger.info(f"Query processed successfully with {len(source_files)} source files found")
            
            return {
                'response': response_text,
                'sources': source_files
            }

            logger.info("Query processed successfully")
            return response.text

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

    def direct_retrieval_query(self, query_text: str, top_k: int = 15, vector_distance_threshold: float = 0.8):
        """Perform direct context retrieval using new syntax"""
        if not self.rag_corpus:
            raise Exception("No corpus available for retrieval")

        try:
            logger.info(f"Performing direct retrieval for: {query_text[:100]}...")

            # Direct context retrieval using new syntax
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=top_k,  # Optional
                filter=rag.Filter(vector_distance_threshold=vector_distance_threshold),  # Optional
            )

            response = rag.retrieval_query(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=self.rag_corpus.name,
                        # Optional: supply IDs from `rag.list_files()`.
                        # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                    )
                ],
                text=query_text,
                rag_retrieval_config=rag_retrieval_config,
            )

            logger.info("Direct retrieval completed successfully")
            return response

        except Exception as e:
            logger.error(f"Error in direct retrieval: {str(e)}")
            raise

# Initialize the RAG system
rag_system = RAGSystem()


def initialize_rag_system():
    """Initialize the RAG system with current Vertex AI API"""
    global rag_system

    try:
        # Initialize Vertex AI and create/load corpus
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
        'has_corpus': rag_system.rag_corpus is not None,
        'has_model': rag_system.rag_model is not None,
        'has_retrieval_tool': rag_system.rag_retrieval_tool is not None,
        'corpus_name': rag_system.rag_corpus.name if rag_system.rag_corpus else None
    })



# Update your query endpoint
@app.route('/query', methods=['POST'])
def query():
    """Process a research query with auto-recovery and source links"""
    global rag_system

    logger.info(f"Query endpoint called. Initialized: {rag_system.initialized}")

    try:
        data = request.get_json()
        logger.info(f"Received data: {data}")

        if not data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400

        user_query = data.get('query', '').strip()
        logger.info(f"User query: {user_query}")

        if not user_query:
            logger.error("Empty query received")
            return jsonify({'error': 'Query cannot be empty'}), 400

        # Process the query using RAG system with sources
        result = rag_system.query_with_sources(user_query)

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
    """Enhanced health check endpoint for Cloud Run"""
    is_healthy = rag_system.health_check()

    return jsonify({
        'status': 'healthy' if is_healthy else 'unhealthy',
        'initialized': rag_system.initialized,
        'has_corpus': rag_system.rag_corpus is not None,
        'corpus_name': rag_system.rag_corpus.name if rag_system.rag_corpus else None,
        'auto_recovery': is_healthy
    }), 200 if is_healthy else 503


@app.route('/test')
def test():
    """Simple test endpoint with health check"""
    is_healthy = rag_system.health_check()

    return jsonify({
        'message': 'Flask server is working!',
        'initialized': rag_system.initialized,
        'healthy': is_healthy,
        'has_corpus': rag_system.rag_corpus is not None,
        'corpus_name': rag_system.rag_corpus.name if rag_system.rag_corpus else None
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

# #
# # @app.route('/create-corpus', methods=['POST'])
# # def create_corpus():
# #     """Create a new RAG corpus"""
# #     try:
# #         data = request.get_json()
# #         display_name = data.get('display_name', 'FTSC Research Papers')
# #         paths = data.get('paths', [])
# #
# #         logger.info(f"Creating corpus: {display_name}")
# #         success = rag_system.create_corpus(display_name, paths)
# #
# #         if success:
# #             # Setup the model after creating corpus
# #             model_success = rag_system.setup_model()
# #             if model_success:
# #                 rag_system.initialized = True
# #             return jsonify({
# #                 'success': model_success,
# #                 'message': 'Corpus created and model setup completed' if model_success else 'Corpus created but model setup failed',
# #                 'corpus_name': rag_system.rag_corpus.name if rag_system.rag_corpus else None
# #             })
# #         else:
# #             return jsonify({
# #                 'success': False,
# #                 'message': 'Failed to create corpus'
# #             })
# #
# #     except Exception as e:
# #         logger.error(f"Error creating corpus: {str(e)}")
# #         return jsonify({
# #             'success': False,
# #             'message': f'Error creating corpus: {str(e)}'
# #         })
# #
# # @app.route('/load-corpus', methods=['POST'])
# # def load_corpus():
# #     """Load an existing RAG corpus"""
# #     try:
# #         data = request.get_json()
# #         corpus_name = data.get('corpus_name')
# #
# #         if not corpus_name:
# #             return jsonify({
# #                 'success': False,
# #                 'message': 'Corpus name is required'
# #             })
# #
# #         logger.info(f"Loading corpus: {corpus_name}")
# #         success = rag_system.load_existing_corpus(corpus_name)
# #
# #         if success:
# #             # Setup the model after loading corpus
# #             model_success = rag_system.setup_model()
# #             if model_success:
# #                 rag_system.initialized = True
# #             return jsonify({
# #                 'success': model_success,
# #                 'message': 'Corpus loaded and model setup completed' if model_success else 'Corpus loaded but model setup failed',
# #                 'corpus_name': rag_system.rag_corpus.name if rag_system.rag_corpus else None
# #             })
# #         else:
# #             return jsonify({
# #                 'success': False,
# #                 'message': 'Failed to load corpus'
# #             })
# #
# #     except Exception as e:
# #         logger.error(f"Error loading corpus: {str(e)}")
# #         return jsonify({
# #             'success': False,
# #             'message': f'Error loading corpus: {str(e)}'
# #         })
# #
@app.route('/test-corpus', methods=['GET'])
def test_corpus():
    """Test if corpus has documents and show corpus info"""
    try:
        # Perform health check first
        is_healthy = rag_system.health_check()

        if rag_system.rag_corpus:
            return jsonify({
                'corpus_name': rag_system.rag_corpus.name,
                'display_name': rag_system.rag_corpus.display_name if hasattr(rag_system.rag_corpus,
                                                                              'display_name') else 'N/A',
                'status': 'Corpus exists and loaded',
                'healthy': is_healthy,
                'initialized': rag_system.initialized
            })
        else:
            return jsonify({
                'error': 'No corpus found',
                'healthy': is_healthy,
                'initialized': rag_system.initialized
            })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'healthy': False,
            'initialized': rag_system.initialized
        })


@app.route('/corpus-info', methods=['GET'])
def corpus_info():
    """Get detailed corpus information"""
    try:
        if not rag_system.health_check():
            return jsonify({'error': 'System not healthy'}), 503

        if rag_system.rag_corpus:
            info = {
                'corpus_name': rag_system.rag_corpus.name,
                'status': 'Active',
                'system_initialized': rag_system.initialized,
                'has_model': rag_system.rag_model is not None,
                'has_retrieval_tool': rag_system.rag_retrieval_tool is not None
            }

            # Try to get additional corpus info if available
            try:
                if hasattr(rag_system.rag_corpus, 'display_name'):
                    info['display_name'] = rag_system.rag_corpus.display_name
            except:
                pass

            return jsonify(info)
        else:
            return jsonify({'error': 'No corpus available'}), 404

    except Exception as e:
        logger.error(f"Error getting corpus info: {str(e)}")
        return jsonify({'error': f'Error getting corpus info: {str(e)}'}), 500

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






