import streamlit as st
import os
from typing import List, Tuple, Optional
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool, GenerationResponse, GenerationConfig
import vertexai
import time

# Configuration
PROJECT_ID = "ftsc-rag-demo"
LOCATION = "us-central1"

import json
import tempfile
from google.oauth2 import service_account

# def setup_authentication():
#     """Setup Google Cloud authentication via file upload"""
#     uploaded_file = st.sidebar.file_uploader(
#         "Upload Google Cloud Service Account Key (JSON)", 
#         type="json",
#         help="Upload your service account key file to authenticate with Google Cloud"
#     )
    
#     if uploaded_file is not None:
#         # Save the uploaded file temporarily
#         with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
#             credentials_dict = json.load(uploaded_file)
#             json.dump(credentials_dict, f)
#             temp_path = f.name
        
#         # Set the environment variable
#         os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = temp_path
#         st.sidebar.success("✅ Credentials uploaded successfully!")
#         return True
    
#     # Check if credentials are already set
#     if os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
#         st.sidebar.success("✅ Credentials already configured")
#         return True
    
#     st.sidebar.warning("⚠️ Please upload your service account key to continue")
#     return False

class RAGPipeline:
    def __init__(self):
        self.rag_corpus = None
        self.rag_model = None
        self.initialized = False
        
    def initialize_vertex_ai(self):
        """Initialize Vertex AI with project configuration"""
        try:
            # Check for authentication
            service_account_info = st.secrets["gcp_service_account"]
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(dict(service_account_info), f)
                service_account_path = f.name
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_account_path
            
            if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
                st.error("Google Cloud credentials not found. Please authenticate first.")
                return False
                
            vertexai.init(project=PROJECT_ID, location="us-central1")
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
               llm_model_name: str = "gemini-2.0-flash-001", temperature: float = 1.0):
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
            generation_config = GenerationConfig(temperature=temperature)
            self.rag_model = GenerativeModel(
                model_name=llm_model_name, 
                tools=[rag_retrieval_tool],
                generation_config=generation_config
            )
            
            self.initialized = True
            return True
            
        except Exception as e:
            st.error(f"Failed to setup model: {str(e)}")
            return False
    
    # def query(self, user_query: str, system_prompt: str) -> Optional[GenerationResponse]:
        # """Generate response for user query"""
        # if not self.initialized:
            # st.error("RAG pipeline not initialized")
            # return None
            
        # try:
            # full_query = system_prompt + user_query
            # response = self.rag_model.generate_content(full_query)
            # return response
        # except Exception as e:
            # st.error(f"Failed to generate response: {str(e)}")
            # return None
    
    def query(self, user_query: str, system_prompt: str, chat_history: List[Tuple[str, str]] = None) -> Optional[GenerationResponse]:
        """Generate response for user query with conversation context"""
        if not self.initialized:
            st.error("RAG pipeline not initialized")
            return None
            
        try:
            # Build conversation context
            conversation_context = system_prompt
            
            if chat_history:
                conversation_context += "\n\nPrevious conversation:\n"
                for prev_query, prev_response in chat_history[-3:]:  # Use last 3 exchanges
                    conversation_context += f"User: {prev_query}\nAssistant: {prev_response}\n\n"
            
            conversation_context += f"Current user query: {user_query}"
            
            response = self.rag_model.generate_content(conversation_context)
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
        page_title="Flight Test Safety Council RAG Pipeline",
        page_icon="🔍",
        layout="wide"
    )
    
    # Add authentication check
    # if not setup_authentication():
    #     st.info("👈 Please upload your Google Cloud service account key in the sidebar to get started.")
    #     return
    
    st.title("🔍 Flight Test Safety Committee - AI Search Tool")
    st.markdown("*AI-powered search assistant for the Flight Test Safety Database*")
    
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
        # RAG Parameters
        st.subheader("RAG Parameters")
        top_k = st.slider(
            "Number of sources to retrieve", 
            3, 15, 10,
            help="Controls how many relevant documents the model looks at when compiling an answer to your query."
        )
        vector_threshold = st.slider(
            "Vector distance threshold", 
            0.1, 1.0, 0.4, 0.1,
            help="Sets the minimum similarity required for a document to be considered relevant (lower = more strict)."
        )
        temperature = st.slider(
            "Model temperature", 
            0.0, 2.0, 1.0, 0.1,
            help="Controls creativity of responses (0.0 = deterministic, 1.0 = default, 2.0 = very creative)."
        )
        
        # Model selection
        # model_options = [
            # "gemini-2.0-flash-001",
            # "gemini-1.5-pro",
            # "gemini-1.5-flash"
        # ]
        selected_model = "gemini-2.0-flash-001" #st.selectbox("LLM Model", model_options)
        
        # Data source configuration
        #st.subheader("Data Sources")
        # default_path = "https://drive.google.com/drive/folders/1UZlVFT1aIDTD3J42wL-0Rn9BFwZDOJlD"
        # data_paths = st.text_area(
            # "Google Drive folder URLs (one per line):",
            # value=default_path,
            # help="Enter Google Drive folder URLs containing your research papers"
        # )
        
        # Parse paths
        #paths = [path.strip() for path in data_paths.split('\n') if path.strip()]
        
        # Corpus management
        #st.subheader("Corpus Management")
        #corpus_name = st.text_input("Corpus Name", "demo_corpus")
        
        if st.button("🚀 Initialize RAG Pipeline", type="primary"):
            # Hardcoded values
            corpus_name = "FTSC Database"
            paths = ["https://drive.google.com/drive/folders/1UZlVFT1aIDTD3J42wL-0Rn9BFwZDOJlD"]
            
            with st.spinner("Initializing RAG pipeline..."):
                # Initialize Vertex AI
                if st.session_state.rag_pipeline.initialize_vertex_ai():
                    st.success("✅ Vertex AI initialized")
                    
                    # Create corpus
                    if st.session_state.rag_pipeline.create_corpus(corpus_name, paths):
                        st.success("✅ Corpus created and files imported")
                        
                        # Setup model
                        if st.session_state.rag_pipeline.setup_model(top_k=top_k, vector_distance_threshold=vector_threshold, llm_model_name=selected_model, temperature=temperature):
                            st.success("✅ RAG model ready!")
                            st.session_state.corpus_created = True
                            st.rerun()
                            
        # if st.button("🚀 Initialize RAG Pipeline", type="primary"):
            # if not paths:
                # st.error("Please provide at least one data source path")
            # else:
                # with st.spinner("Initializing RAG pipeline..."):
                    # # Initialize Vertex AI
                    # if st.session_state.rag_pipeline.initialize_vertex_ai():
                        # st.success("✅ Vertex AI initialized")
                        
                        # # Create corpus
                        # if st.session_state.rag_pipeline.create_corpus(corpus_name, paths):
                            # st.success("✅ Corpus created and files imported")
                            
                            # # Setup model
                            # if st.session_state.rag_pipeline.setup_model(
                                # top_k, vector_threshold, selected_model
                            # ):
                                # st.success("✅ RAG model ready!")
                                # st.session_state.corpus_created = True
                                # st.rerun()
        
        # Show advanced options
        show_chunks = st.checkbox("Show retrieved chunks", False)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main content area
    
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
    
    if not st.session_state.corpus_created:
        st.info("Please initialize the RAG pipeline using the sidebar to get started.")
        
        # Show demo information
        st.markdown(f"""
        This tool is a Retrieval-Augmented Generation (RAG)-based intelligent search tool for the Flight Test Safety Committee paper database:
        
        - **Technical Search**: Find relevant papers using semantic search across flight test documentation
        - **Lessons Learned**: Get detailed insights based on prior flight test experiences and safety data
        - **Relevance Ranking**: Papers ranked by relevance to your specific flight test scenario
        - **Source Attribution**: Clear citations from FTSC papers and technical reports
        
        **To use:**
        1. Configure your search parameters in the sidebar (or use the default values)
        2. Click "Initialize RAG pipeline" to connect to the FTSC database
        3. Ask questions about flight test procedures, safety considerations, or lessons learned
        4. Get comprehensive answers with source citations
        
        **Example queries:**
        - "What are some prior papers about high altitude flight testing?"
        - "What do I need to know about autonomous vehicle flight testing?"
        - "What safety considerations apply to envelope expansion testing?"
        - "Are there lessons learned from flutter testing incidents?"
        
        ## How does this tool work?
        
        This tool uses a technique called **Retrieval-Augmented Generation**, which uses a large language model(LLM) connected to a database of information (a "corpus").

        Retrieval-Augmented Generation (RAG) is a technique to fine-tune an LLM to a specific database or use case without the need for retraining, which would be expensive and infeasible for small-scale use. 
        
        RAG does the following:
            
        1. **Corpus Creation**: Upon setup, the files in the database (corpus) are broken into "chunks" which are converted into numerical vectors ("embeddings") that encode their semantic meaning. From this point on, the RAG tool works with this vector database of document embeddings, not the raw files (e.g. PDFs) in the original database.

        2. **Document Retrieval**: When you ask a query, it is converted into a vector embedding in the same way as the database files. The **retrieval system** then searches through embeddings of all documents in the FTSC database to find the vectors in the corpus that are most similar to the query - these correspond to the most relevant papers and their specific relevant sections.

        3. **Context Assembly**: The most relevant document chunks are retrieved and combined with your original question to create a comprehensive context.

        4. **Response Generation**: A large language model (in our case, a lightweight variant of Gemini) uses the combined context from step 3 to generate a response to your query. 
        
        Because the context from step 3 is focused on the most relevant content from the database, the response is tailored to your query and is less likely to be distracted by irrelevant content in the database.
        
        Like other LLM-based chat tools (e.g. ChatGPT, Claude, Gemini), this tool uses a **system prompt** which your query is appended to. This prompt shapes the model's behavior, tone, and things it is allowed and not allowed to say in response to your query. In our case, we use this system prompt:
        """)
        
        st.code(system_prompt, language="text")
        
        
    else:
        # Chat interface

        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (query, response) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(query)
                with st.chat_message("assistant"):
                    st.write(response)

        # Query input at the bottom
        user_query = st.chat_input("Ask about flight test techniques, safety considerations, lessons learned...")

        if user_query:
            # Add user message to chat immediately
            st.session_state.chat_history.append((user_query, ""))
            
            # Display user message
            with st.chat_message("user"):
                st.write(user_query)
            
            # Generate and display response
            with st.chat_message("assistant"):
                with st.spinner("Searching and generating response..."):
                    # Pass chat history for context (excluding the current incomplete entry)
                    response = st.session_state.rag_pipeline.query(
                        user_query, 
                        system_prompt, 
                        st.session_state.chat_history[:-1]  # Exclude current incomplete entry
                    )
                    
                    if response:
                        st.write(response.text)
                        
                        # Update the last entry with the response
                        st.session_state.chat_history[-1] = (user_query, response.text)
                        
                        # Show retrieved chunks if requested
                        if show_chunks:
                            with st.expander("🔍 Retrieved Chunks (Debug Info)"):
                                chunks = st.session_state.rag_pipeline.get_retrieved_chunks(
                                    user_query, system_prompt, top_k, vector_threshold
                                )
                                if chunks:
                                    st.code(str(chunks), language="text")
                    else:
                        error_msg = "Sorry, I couldn't generate a response. Please try again."
                        st.error(error_msg)
                        st.session_state.chat_history[-1] = (user_query, error_msg)
            
            st.rerun()
        # st.subheader("💬 Search Assistant")
        
        # # Display chat history
        # for i, (query, response) in enumerate(st.session_state.chat_history):
            # with st.container():
                # st.markdown(f"**You:** {query}")
                # st.markdown(f"**Assistant:** {response}")
                # st.divider()
        
        # # Query input
        # user_query = st.text_input(
            # "Please enter your query:",
            # placeholder="e.g., What do I need to know about high-altitude flight test?"
        # )
        
        # if st.button("🔍 Search") and user_query:
            # with st.spinner("Searching and generating response..."):
                # # Get response
                # response = st.session_state.rag_pipeline.query(user_query, system_prompt)
                
                # if response:
                    # # Display response
                    # st.markdown("### Response:")
                    # st.markdown(response.text)
                    
                    # # Add to chat history
                    # st.session_state.chat_history.append((user_query, response.text))
                    
                    # # Show retrieved chunks if requested
                    # if show_chunks:
                        # with st.expander("🔍 Retrieved Chunks (Debug Info)"):
                            # chunks = st.session_state.rag_pipeline.get_retrieved_chunks(
                                # user_query, system_prompt, top_k, vector_threshold
                            # )
                            # if chunks:
                                # st.code(str(chunks), language="text")
                    
                    # st.rerun()

if __name__ == "__main__":
    main()
