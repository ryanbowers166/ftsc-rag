import streamlit as st
import os
from typing import List, Tuple, Optional
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool, GenerationResponse
import vertexai
import time

# Configuration
PROJECT_ID = "ftsc-rag-demo"
LOCATION = "us-central1"

import json
import tempfile
from google.oauth2 import service_account

def setup_authentication():
    """Setup Google Cloud authentication via file upload"""
    uploaded_file = st.sidebar.file_uploader(
        "Upload Google Cloud Service Account Key (JSON)", 
        type="json",
        help="Upload your service account key file to authenticate with Google Cloud"
    )
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            credentials_dict = json.load(uploaded_file)
            json.dump(credentials_dict, f)
            temp_path = f.name
        
        # Set the environment variable
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path
        st.sidebar.success("✅ Credentials uploaded successfully!")
        return True
    
    # Check if credentials are already set
    if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
        st.sidebar.success("✅ Credentials already configured")
        return True
    
    st.sidebar.warning("⚠️ Please upload your service account key to continue")
    return False

class RAGPipeline:
    def __init__(self):
        self.rag_corpus = None
        self.rag_model = None
        self.initialized = False
        
    def initialize_vertex_ai(self):
    """Initialize Vertex AI with project configuration"""
    try:
        # Check for authentication
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            st.error("Google Cloud credentials not found. Please authenticate first.")
            return False
            
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        return True
    except Exception as e:
        st.error(f"Failed to initialize Vertex AI: {str(e)}")
        return False
    
    def create_corpus(self, display_name: str, paths: List[str]) -> bool:
        """Create RAG corpus and import files"""
        try:
            # Create embedding model config
            embedding_model_config = rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            )
            
            # Create RagCorpus
            self.rag_corpus = rag.create_corpus(
                display_name=display_name,
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config
                ),
            )
            
            # Import Files to the RagCorpus
            with st.spinner("Importing files to RAG corpus... This may take a few minutes."):
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
            
            return True
            
        except Exception as e:
            st.error(f"Failed to create corpus: {str(e)}")
            return False
    
    def setup_model(self, top_k: int = 7, vector_distance_threshold: float = 0.5, 
                   llm_model_name: str = "gemini-2.0-flash-001"):
        """Setup the RAG model with retrieval tool"""
        try:
            # Direct context retrieval
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=top_k,
                filter=rag.Filter(vector_distance_threshold=vector_distance_threshold),
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
            
            # Create a model instance
            self.rag_model = GenerativeModel(
                model_name=llm_model_name, 
                tools=[rag_retrieval_tool]
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            st.error(f"Failed to setup model: {str(e)}")
            return False
    
    def query(self, user_query: str, system_prompt: str) -> Optional[GenerationResponse]:
        """Generate response for user query"""
        if not self.initialized:
            st.error("RAG pipeline not initialized")
            return None
            
        try:
            full_query = system_prompt + user_query
            response = self.rag_model.generate_content(full_query)
            return response
        except Exception as e:
            st.error(f"Failed to generate response: {str(e)}")
            return None
    
    def get_retrieved_chunks(self, user_query: str, system_prompt: str, 
                           top_k: int = 7, vector_distance_threshold: float = 0.5):
        """Get retrieved chunks for debugging"""
        try:
            rag_retrieval_config = rag.RagRetrievalConfig(
                top_k=top_k,
                filter=rag.Filter(vector_distance_threshold=vector_distance_threshold),
            )
            
            full_query = system_prompt + user_query
            retrieval_response = rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=self.rag_corpus.name)],
                text=full_query,
                rag_retrieval_config=rag_retrieval_config,
            )
            return retrieval_response
        except Exception as e:
            st.error(f"Failed to retrieve chunks: {str(e)}")
            return None

def main():
    st.set_page_config(
        page_title="Research Paper RAG Demo",
        page_icon="🔍",
        layout="wide"
    )
    
    # Add authentication check
    if not setup_authentication():
        st.info("👈 Please upload your Google Cloud service account key in the sidebar to get started.")
        return
    
    st.title("🔍 Research Paper RAG Demo")
    st.markdown("*AI-powered research assistant for technical conference papers*")
    
    # Initialize session state
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    if 'corpus_created' not in st.session_state:
        st.session_state.corpus_created = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # RAG Parameters
        st.subheader("RAG Parameters")
        top_k = st.slider("Number of sources to retrieve", 3, 15, 7)
        vector_threshold = st.slider("Vector distance threshold", 0.1, 1.0, 0.5, 0.1)
        
        # Model selection
        model_options = [
            "gemini-2.0-flash-001",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
        selected_model = st.selectbox("LLM Model", model_options)
        
        # Data source configuration
        st.subheader("Data Sources")
        default_path = "https://drive.google.com/drive/folders/1UZlVFT1aIDTD3J42wL-0Rn9BFwZDOJlD"
        data_paths = st.text_area(
            "Google Drive folder URLs (one per line):",
            value=default_path,
            help="Enter Google Drive folder URLs containing your research papers"
        )
        
        # Parse paths
        paths = [path.strip() for path in data_paths.split('\n') if path.strip()]
        
        # Corpus management
        st.subheader("Corpus Management")
        corpus_name = st.text_input("Corpus Name", "demo_corpus")
        
        if st.button("🚀 Initialize RAG Pipeline", type="primary"):
            if not paths:
                st.error("Please provide at least one data source path")
            else:
                with st.spinner("Initializing RAG pipeline..."):
                    # Initialize Vertex AI
                    if st.session_state.rag_pipeline.initialize_vertex_ai():
                        st.success("✅ Vertex AI initialized")
                        
                        # Create corpus
                        if st.session_state.rag_pipeline.create_corpus(corpus_name, paths):
                            st.success("✅ Corpus created and files imported")
                            
                            # Setup model
                            if st.session_state.rag_pipeline.setup_model(
                                top_k, vector_threshold, selected_model
                            ):
                                st.success("✅ RAG model ready!")
                                st.session_state.corpus_created = True
                                st.rerun()
        
        # Show advanced options
        show_chunks = st.checkbox("Show retrieved chunks", False)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    if not st.session_state.corpus_created:
        st.info("👈 Please initialize the RAG pipeline using the sidebar to get started.")
        
        # Show demo information
        st.markdown("## About This Demo")
        st.markdown("""
        This demo showcases a Retrieval-Augmented Generation (RAG) system for research papers:
        
        - **🔍 Intelligent Search**: Find relevant papers using semantic search
        - **📚 Context-Aware Responses**: Get detailed answers based on paper content
        - **🎯 Relevance Ranking**: Papers ranked by relevance to your query
        - **📖 Source Attribution**: Clear citations and references
        
        **To use:**
        1. Configure your parameters in the sidebar
        2. Add your Google Drive folder URLs containing research papers
        3. Click "Initialize RAG Pipeline" and wait for setup
        4. Start asking questions about the research papers!
        """)
        
    else:
        # System prompt
        system_prompt = """You are a research assistant analyzing technical conference papers. Your task is to identify papers relevant to the specific topic mentioned in the query. When determining relevance:
        1. Focus on direct technical connections to the query topic
        2. Consider both explicit mentions and implicit relevance through related methodologies
        3. Rank papers by how central the query topic is to the paper's main contributions
        4. Be precise about why each paper is or isn't relevant
        5. Cite specific sections when possible
        6. If uncertain about relevance, explain why
        7. Always mention sources by title, not just their source number.
        8. Do not recommend specific courses of action to the user. Only suggest which sources they should read and why.
        Based on these criteria, analyze the provided papers to answer the query: """
        
        # Chat interface
        st.subheader("💬 Research Assistant")
        
        # Display chat history
        for i, (query, response) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {query}")
                st.markdown(f"**Assistant:** {response}")
                st.divider()
        
        # Query input
        user_query = st.text_input(
            "Ask a question about the research papers:",
            placeholder="e.g., What papers discuss transformer architectures for natural language processing?"
        )
        
        if st.button("🔍 Search") and user_query:
            with st.spinner("Searching and generating response..."):
                # Get response
                response = st.session_state.rag_pipeline.query(user_query, system_prompt)
                
                if response:
                    # Display response
                    st.markdown("### Response:")
                    st.markdown(response.text)
                    
                    # Add to chat history
                    st.session_state.chat_history.append((user_query, response.text))
                    
                    # Show retrieved chunks if requested
                    if show_chunks:
                        with st.expander("🔍 Retrieved Chunks (Debug Info)"):
                            chunks = st.session_state.rag_pipeline.get_retrieved_chunks(
                                user_query, system_prompt, top_k, vector_threshold
                            )
                            if chunks:
                                st.code(str(chunks), language="text")
                    
                    st.rerun()

if __name__ == "__main__":
    main()