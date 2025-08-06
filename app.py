import os
import logging
import threading
import time
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool
import vertexai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class RAGService:
    def __init__(self):
        self.rag_model = None
        self.rag_corpus = None
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        self.initialization_error = None
        
        # Cloud Run configuration
        self.project_id = os.environ.get('PROJECT_ID', 'ftsc-rag-demo')
        self.region = os.environ.get('REGION', 'us-central1')
        self.corpus_name = os.environ.get('CORPUS_NAME', 'cloud_run_corpus')
        self.drive_folder_id = os.environ.get('DRIVE_FOLDER_ID', '1UZlVFT1aIDTD3J42wL-0Rn9BFwZDOJlD')
        
        # RAG parameters
        self.top_k = int(os.environ.get('TOP_K', '7'))
        self.vector_distance_threshold = float(os.environ.get('VECTOR_DISTANCE_THRESHOLD', '0.5'))
        self.llm_model_name = os.environ.get('LLM_MODEL', 'gemini-2.0-flash-001')
        
        # Initialize on startup
        self._initialize_async()
        
    def _initialize_async(self):
        """Initialize RAG service asynchronously"""
        def init_worker():
            try:
                self.initialize_rag()
            except Exception as e:
                logger.error(f"Failed to initialize RAG service: {e}")
                self.initialization_error = str(e)
        
        thread = threading.Thread(target=init_worker)
        thread.daemon = True
        thread.start()
        
    def initialize_rag(self):
        """Initialize the RAG system"""
        with self.initialization_lock:
            if self.is_initialized:
                return
                
            try:
                logger.info("Starting RAG initialization...")
                
                # Initialize Vertex AI
                vertexai.init(project=self.project_id, location=self.region)
                
                # Check if corpus already exists
                corpus_name = f"projects/{self.project_id}/locations/{self.region}/ragCorpora/{self.corpus_name}"
                
                try:
                    # Try to get existing corpus
                    self.rag_corpus = rag.get_corpus(name=corpus_name)
                    logger.info(f"Using existing corpus: {corpus_name}")
                except Exception as e:
                    logger.info(f"Creating new corpus: {e}")
                    # Create new corpus
                    embedding_model_config = rag.RagEmbeddingModelConfig(
                        vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                            publisher_model="publishers/google/models/text-embedding-005"
                        )
                    )
                    
                    self.rag_corpus = rag.create_corpus(
                        display_name=self.corpus_name,
                        backend_config=rag.RagVectorDbConfig(
                            rag_embedding_model_config=embedding_model_config
                        ),
                    )
                    
                    # Import files only for new corpus
                    logger.info("Importing files to corpus...")
                    paths = [f"https://drive.google.com/drive/folders/{self.drive_folder_id}"]
                    
                    rag.import_files(
                        self.rag_corpus.name,
                        paths,
                        transformation_config=rag.TransformationConfig(
                            chunking_config=rag.ChunkingConfig(
                                chunk_size=1024,
                                chunk_overlap=150,
                            ),
                        ),
                        max_embedding_requests_per_min=1000,
                    )
                    logger.info("Files imported successfully")
                
                # Create retrieval configuration
                rag_retrieval_config = rag.RagRetrievalConfig(
                    top_k=self.top_k,
                    filter=rag.Filter(vector_distance_threshold=self.vector_distance_threshold),
                )
                
                # Create RAG retrieval tool
                rag_retrieval_tool = Tool.from_retrieval(
                    retrieval=rag.Retrieval(
                        source=rag.VertexRagStore(
                            rag_resources=[rag.RagResource(rag_corpus=self.rag_corpus.name)],
                            rag_retrieval_config=rag_retrieval_config,
                        ),
                    )
                )
                
                # Create model instance
                self.rag_model = GenerativeModel(
                    model_name=self.llm_model_name, 
                    tools=[rag_retrieval_tool]
                )
                
                self.is_initialized = True
                logger.info("RAG service initialized successfully!")
                
            except Exception as e:
                logger.error(f"Error initializing RAG service: {e}")
                self.initialization_error = str(e)
                raise e
    
    def query_rag(self, user_query: str) -> Dict[str, Any]:
        """Process a query through the RAG system"""
        if not self.is_initialized:
            if self.initialization_error:
                return {
                    "success": False,
                    "error": f"RAG service initialization failed: {self.initialization_error}"
                }
            return {
                "success": False,
                "error": "RAG service is still initializing. Please try again in a few moments."
            }
        
        try:
            system_prompt = """You are a helpful chat agent helping a flight test professional analyze technical papers and documentation. You will help the user find relevant sources in the database about flight test techniques, procedures, considerations, and lessons learned.
        
            When the user asks a query, you will return a list of sources in the database that are relevant to that query, along with a 1-paragraph summary of the relevant content from each source. 
            
            When answering questions about specific types of flight testing (e.g., high altitude, autonomous vehicles, supersonic, etc.), focus on:
            1. **Unique characteristics** and challenges specific to that test type
            2. **Specialized equipment, procedures, or methodologies** required
            3. **Specific risks, considerations, or constraints** that don't apply to general flight testing
            4. **Technical differences** from standard flight test approaches
            5. **Specialized certification or regulatory requirements** if applicable
            
            ALWAYS cite specific sources in the database that you use to form your responses. When citing sources, always mention paper titles and explain why each source is relevant to the specific type of testing being discussed.
            
            Avoid generic flight test advice (like "review test cards" or "hold safety briefings") unless it's specifically adapted for the test type in question.
            
            Maintain the conversation to the best of your ability. For example, respond to the user's follow-on questions, and end your responses with your own follow-on questions to continue the conversation.
            
            Query: """
            
            query = system_prompt + user_query
            
            # Generate response with timeout
            response = self.rag_model.generate_content(query)
            
            return {
                "success": True,
                "response": response.text,
                "query": user_query,
                "model": self.llm_model_name
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "success": False,
                "error": f"Query processing failed: {str(e)}",
                "query": user_query
            }

# Initialize the RAG service
rag_service = RAGService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        "status": "healthy",
        "initialized": rag_service.is_initialized,
        "initialization_error": rag_service.initialization_error,
        "project_id": rag_service.project_id,
        "region": rag_service.region
    })

@app.route('/ready', methods=['GET'])
def readiness_check():
    """Readiness check for Cloud Run"""
    if rag_service.is_initialized:
        return jsonify({"status": "ready"}), 200
    else:
        return jsonify({
            "status": "not_ready",
            "error": rag_service.initialization_error
        }), 503

@app.route('/query', methods=['POST'])
def query():
    """Process a RAG query"""
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Missing 'query' parameter"
            }), 400
        
        user_query = data['query']
        
        if not user_query.strip():
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400
        
        # Log the query for monitoring
        logger.info(f"Processing query: {user_query[:100]}...")
        
        # Process the query
        result = rag_service.query_rag(user_query)
        
        if result.get('success'):
            logger.info("Query processed successfully")
            return jsonify(result)
        else:
            logger.error(f"Query failed: {result.get('error')}")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return jsonify({
            "success": False,
            "error": f"Internal server error: {str(e)}"
        }), 500

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "service": "RAG Research Assistant",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "query": "/query (POST)"
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)