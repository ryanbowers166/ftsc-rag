
# =============================================================================
# RAG IMPORT COMPATIBILITY LAYER - Add this right after your basic imports
# =============================================================================
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import logging
from typing import List, Tuple, Optional
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool
from google.cloud import aiplatform
import json
import tempfile
from vertexai.rag import VertexPredictionEndpoint

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try different import paths for RAG
rag = None
rag_import_success = False

# Pattern 1: Try direct import (official way)
try:
    from vertexai import rag
    logger.info("✅ SUCCESS: Imported RAG from vertexai (official path)")
    rag_import_success = True
except ImportError as e:
    logger.warning(f"❌ Failed to import from vertexai: {e}")

# Pattern 2: Try preview import (fallback)
if not rag_import_success:
    try:
        from vertexai.preview import rag
        logger.info("✅ SUCCESS: Imported RAG from vertexai.preview (fallback path)")
        rag_import_success = True
    except ImportError as e:
        logger.warning(f"❌ Failed to import from vertexai.preview: {e}")

# Pattern 3: Try using aiplatform directly (alternative approach)
if not rag_import_success:
    try:
        from google.cloud.aiplatform import rag
        logger.info("✅ SUCCESS: Imported RAG from google.cloud.aiplatform (alternative path)")
        rag_import_success = True
    except ImportError as e:
        logger.warning(f"❌ Failed to import from google.cloud.aiplatform: {e}")

# If all imports failed, create a fallback that will show clear error messages
if not rag_import_success:
    logger.error("❌ All RAG import attempts failed!")
    
    # Create a mock rag module that will give helpful error messages
    class MockRAG:
        def __getattr__(self, name):
            raise ImportError(f"""
            RAG module could not be imported. This might be because:
            1. You need to update vertexai: pip install --upgrade vertexai google-cloud-aiplatform
            2. The RAG API might not be available in your region
            3. You might need to enable the RAG API in your project
            
            Attempted to access: {name}
            Please check your installation and try again.
            """)
    
    rag = MockRAG()
    logger.error("Created mock RAG module - the app will show clear error messages")

# Verify key RAG components are available
if rag_import_success:
    logger.info("Testing key RAG components...")
    key_components = [
        'create_corpus', 'import_files', 'RagEmbeddingModelConfig', 
        'VertexPredictionEndpoint', 'RagVectorDbConfig', 'TransformationConfig',
        'ChunkingConfig', 'RagRetrievalConfig', 'Filter', 'retrieval_query',
        'RagResource', 'Retrieval', 'VertexRagStore'
    ]
    
    missing_components = []
    for component in key_components:
        if hasattr(rag, component):
            logger.info(f"   ✅ {component}")
        else:
            logger.warning(f"   ❌ {component} NOT FOUND")
            missing_components.append(component)
    
    if missing_components:
        logger.warning(f"Missing RAG components: {missing_components}")
        logger.warning("Some functionality may not work correctly")

logger.info("RAG import compatibility layer setup complete")


#from flask import Flask, request, jsonify, render_template_string
#from flask_cors import CORS
#import os
#import logging
#from typing import List, Tuple, Optional
#import vertexai
#from vertexai.generative_models import GenerativeModel, GenerationConfig, Tool
#from vertexai.preview import rag
#from google.cloud import aiplatform
#import json
#import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for RAG system
class RAGSystem:
    def __init__(self):
        self.rag_model = None
        self.rag_corpus = None
        self.rag_retrieval_tool = None
        self.initialized = False
        self.PROJECT_ID = "ftscrag"
        self.LOCATION = "us-central1"
        
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
        
    def initialize(self):
        """Initialize the RAG system with Vertex AI"""
        try:
            if not self.setup_authentication():
                logger.error("Authentication setup failed")
                return False
            
            logger.info("Initializing Vertex AI...")
            # Initialize Vertex AI once per session
            vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)
            aiplatform.init(project=self.PROJECT_ID, location=self.LOCATION)
            
            # Try to create corpus with Google Drive folder
            success = self.create_corpus_from_drive()
            if success:
                # Setup the model after creating corpus
                model_success = self.setup_model()
                if model_success:
                    self.initialized = True
                    logger.info("RAG system initialized successfully!")
                    return True
                else:
                    logger.error("Failed to setup model after corpus creation")
                    return False
            else:
                logger.error("Failed to create corpus from Google Drive")
                return False
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {str(e)}")
            return False
    
    def create_corpus_from_drive(self) -> bool:
        """Create RAG corpus from Google Drive folder using new syntax"""
        try:
            logger.info("Creating corpus from Google Drive folder...")
            
            # Google Drive folder URL
            drive_folder_url = "https://drive.google.com/drive/folders/1Qif8tvURTHOOrtrosTQ4YU077yPnuiTB"
            
            # Configure embedding model using new syntax
            embedding_model_config = rag.EmbeddingModelConfig(
                vertex_prediction_endpoint=VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            )
            
            # Create RagCorpus using new syntax
            self.rag_corpus = rag.create_corpus(
                display_name="FTSC Research Papers Corpus",
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
                        chunk_overlap=100,
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
    
    def create_corpus(self, display_name: str, paths: List[str]) -> bool:
        """Create RAG corpus and import files using new syntax"""
        try:
            logger.info(f"Creating corpus: {display_name}")
            
            # Configure embedding model using new syntax
            embedding_model_config = rag.EmbeddingModelConfig(
                vertex_prediction_endpoint=VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            )
            
            # Create RagCorpus using new syntax
            self.rag_corpus = rag.create_corpus(
                display_name=display_name,
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config
                ),
            )
            
            logger.info(f"Corpus created: {self.rag_corpus.name}")
            
            # Import files using new syntax
            if paths:
                logger.info("Importing files to RAG corpus...")
                rag.import_files(
                    self.rag_corpus.name,
                    paths,
                    # Optional transformation config
                    transformation_config=rag.TransformationConfig(
                        chunking_config=rag.ChunkingConfig(
                            chunk_size=512,
                            chunk_overlap=100,
                        ),
                    ),
                    max_embedding_requests_per_min=1000,  # Optional
                )
                logger.info("Files imported successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create corpus: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
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
    
    def setup_model(self, top_k: int = 3, vector_distance_threshold: float = 0.5, 
                   llm_model_name: str = "gemini-2.0-flash-001", temperature: float = 1.0):
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
    
    def direct_retrieval_query(self, query_text: str, top_k: int = 3, vector_distance_threshold: float = 0.5):
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
    
    def query(self, user_query: str) -> str:
        """Process a query using the RAG system with new syntax"""
        if not self.initialized or not self.rag_model:
            raise Exception("RAG system not initialized")
        
        try:
            logger.info(f"Processing query: {user_query[:100]}...")
            
            # Create a research-focused system prompt
            system_prompt = """You are a research assistant analyzing technical conference papers and documents. 
            Use the retrieval tool to find relevant information from the corpus to answer the user's question.
            
            Guidelines:
            1. Always use the retrieval tool to search for relevant documents first
            2. Base your response on the retrieved information
            3. If you find relevant papers or documents, cite them appropriately
            4. Focus on technical accuracy and provide specific details
            5. If no relevant information is found, clearly state this
            6. Structure your response clearly with key findings and recommendations
            
            User Query: """
            
            full_query = system_prompt + user_query
            
            # Generate response using the RAG model with new syntax
            response = self.rag_model.generate_content(full_query)
            
            logger.info("Query processed successfully")
            return response.text
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            raise

# Initialize the RAG system
rag_system = RAGSystem()

def initialize_rag_system():
    """Initialize the RAG system with current Vertex AI API"""
    global rag_system
    
    try:
        # Initialize Vertex AI and create corpus from Google Drive
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
    """Check if the RAG system is initialized"""
    return jsonify({
        'initialized': rag_system.initialized,
        'has_corpus': rag_system.rag_corpus is not None,
        'has_model': rag_system.rag_model is not None,
        'has_retrieval_tool': rag_system.rag_retrieval_tool is not None
    })

@app.route('/query', methods=['POST'])
def query():
    """Process a research query"""
    global rag_system
    
    logger.info(f"Query endpoint called. Initialized: {rag_system.initialized}")
    
    if not rag_system.initialized or not rag_system.rag_model:
        logger.error("RAG system not initialized")
        return jsonify({'error': 'RAG system not initialized. Please initialize the system first.'}), 500
    
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
        
        # Process the query using RAG system
        response_text = rag_system.query(user_query)
        
        result = {'response': response_text}
        logger.info(f"Returning result with {len(response_text)} characters")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

@app.route('/direct-retrieval', methods=['POST'])
def direct_retrieval():
    """Perform direct context retrieval without generation"""
    global rag_system
    
    if not rag_system.rag_corpus:
        return jsonify({'error': 'No corpus available for retrieval'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400
            
        query_text = data.get('query', '').strip()
        top_k = data.get('top_k', 3)
        vector_distance_threshold = data.get('vector_distance_threshold', 0.5)
        
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
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy', 
        'initialized': rag_system.initialized,
        'has_corpus': rag_system.rag_corpus is not None
    })

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({
        'message': 'Flask server is working!', 
        'initialized': rag_system.initialized,
        'has_corpus': rag_system.rag_corpus is not None
    })

@app.route('/initialize', methods=['POST'])
def manual_initialize():
    """Manual initialization endpoint"""
    logger.info("Manual initialization requested")
    success = initialize_rag_system()
    return jsonify({
        'success': success,
        'initialized': rag_system.initialized,
        'has_corpus': rag_system.rag_corpus is not None,
        'message': 'Initialization successful' if success else 'Initialization failed. Check logs for details.'
    })

@app.route('/create-corpus', methods=['POST'])
def create_corpus():
    """Create a new RAG corpus"""
    try:
        data = request.get_json()
        display_name = data.get('display_name', 'FTSC Research Papers')
        paths = data.get('paths', [])
        
        logger.info(f"Creating corpus: {display_name}")
        success = rag_system.create_corpus(display_name, paths)
        
        if success:
            # Setup the model after creating corpus
            model_success = rag_system.setup_model()
            if model_success:
                rag_system.initialized = True
            return jsonify({
                'success': model_success,
                'message': 'Corpus created and model setup completed' if model_success else 'Corpus created but model setup failed',
                'corpus_name': rag_system.rag_corpus.name if rag_system.rag_corpus else None
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to create corpus'
            })
            
    except Exception as e:
        logger.error(f"Error creating corpus: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error creating corpus: {str(e)}'
        })

@app.route('/load-corpus', methods=['POST'])
def load_corpus():
    """Load an existing RAG corpus"""
    try:
        data = request.get_json()
        corpus_name = data.get('corpus_name')
        
        if not corpus_name:
            return jsonify({
                'success': False,
                'message': 'Corpus name is required'
            })
        
        logger.info(f"Loading corpus: {corpus_name}")
        success = rag_system.load_existing_corpus(corpus_name)
        
        if success:
            # Setup the model after loading corpus
            model_success = rag_system.setup_model()
            if model_success:
                rag_system.initialized = True
            return jsonify({
                'success': model_success,
                'message': 'Corpus loaded and model setup completed' if model_success else 'Corpus loaded but model setup failed',
                'corpus_name': rag_system.rag_corpus.name if rag_system.rag_corpus else None
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to load corpus'
            })
            
    except Exception as e:
        logger.error(f"Error loading corpus: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error loading corpus: {str(e)}'
        })

if __name__ == '__main__':
    # Initialize RAG system on startup
    logger.info("Starting Flask app and initializing RAG system...")
    initialize_rag_system()
    
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)





