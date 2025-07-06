import streamlit as st
import os
from typing import List, Tuple
from vertexai import rag
from vertexai.generative_models import GenerativeModel, Tool, GenerationResponse
import vertexai

# Page config
st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'rag_model' not in st.session_state:
    st.session_state.rag_model = None
if 'rag_corpus' not in st.session_state:
    st.session_state.rag_corpus = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system (cached to avoid re-initialization)"""
    PROJECT_ID = "ftsc-rag-demo"
    display_name = "demo_corpus"
    paths = ["https://drive.google.com/drive/folders/1UZlVFT1aIDTD3J42wL-0Rn9BFwZDOJlD"]
    
    # Initialize Vertex AI
    vertexai.init(project=PROJECT_ID, location="us-central1")
    
    # Create or get existing corpus
    try:
        # Try to list existing corpora first
        existing_corpora = rag.list_corpora()
        rag_corpus = None
        for corpus in existing_corpora:
            if corpus.display_name == display_name:
                rag_corpus = corpus
                break
        
        if not rag_corpus:
            # Create new corpus
            embedding_model_config = rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            )
            
            rag_corpus = rag.create_corpus(
                display_name=display_name,
                backend_config=rag.RagVectorDbConfig(
                    rag_embedding_model_config=embedding_model_config
                ),
            )
            
            # Import files
            rag.import_files(
                rag_corpus.name,
                paths,
                transformation_config=rag.TransformationConfig(
                    chunking_config=rag.ChunkingConfig(
                        chunk_size=1024,
                        chunk_overlap=150,
                    ),
                ),
                max_embedding_requests_per_min=1000,
            )
        
        # Create retrieval tool
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=7,
            filter=rag.Filter(vector_distance_threshold=0.5),
        )
        
        rag_retrieval_tool = Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
                    rag_retrieval_config=rag_retrieval_config,
                ),
            )
        )
        
        # Create model
        rag_model = GenerativeModel(
            model_name="gemini-2.0-flash-001", 
            tools=[rag_retrieval_tool]
        )
        
        return rag_model, rag_corpus
        
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None, None

def main():
    st.title("🔍 Research Assistant RAG Demo")
    st.markdown("Ask questions about technical conference papers in our database!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        show_chunks = st.checkbox("Show retrieved chunks", value=False)
        
        if st.button("Initialize/Reinitialize RAG System"):
            with st.spinner("Initializing RAG system..."):
                st.session_state.rag_model, st.session_state.rag_corpus = initialize_rag_system()
                if st.session_state.rag_model:
                    st.session_state.initialized = True
                    st.success("RAG system initialized!")
                else:
                    st.error("Failed to initialize RAG system")
    
    # Initialize on first run
    if not st.session_state.initialized:
        with st.spinner("Initializing RAG system for first use..."):
            st.session_state.rag_model, st.session_state.rag_corpus = initialize_rag_system()
            if st.session_state.rag_model:
                st.session_state.initialized = True
                st.success("RAG system ready!")
    
    # Main query interface
    if st.session_state.initialized and st.session_state.rag_model:
        # Query input
        user_query = st.text_area(
            "Enter your research question:",
            placeholder="e.g., What papers discuss neural network optimization techniques?",
            height=100
        )
        
        if st.button("Search Papers", type="primary") and user_query:
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
            
            full_query = system_prompt + user_query
            
            # Show retrieved chunks if requested
            if show_chunks:
                st.subheader("Retrieved Document Chunks")
                with st.spinner("Retrieving relevant chunks..."):
                    try:
                        rag_retrieval_config = rag.RagRetrievalConfig(
                            top_k=7,
                            filter=rag.Filter(vector_distance_threshold=0.5),
                        )
                        
                        retrieval_response = rag.retrieval_query(
                            rag_resources=[rag.RagResource(rag_corpus=st.session_state.rag_corpus.name)],
                            text=full_query,
                            rag_retrieval_config=rag_retrieval_config,
                        )
                        st.code(str(retrieval_response), language="text")
                    except Exception as e:
                        st.error(f"Error retrieving chunks: {str(e)}")
            
            # Generate response
            st.subheader("Research Assistant Response")
            with st.spinner("Analyzing papers and generating response..."):
                try:
                    response = st.session_state.rag_model.generate_content(full_query)
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        
        # Example queries
        st.subheader("Example Queries")
        examples = [
            "What papers discuss transformer architectures?",
            "Which papers focus on computer vision applications?",
            "What research covers reinforcement learning algorithms?",
            "Papers about natural language processing methods"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            if cols[i % 2].button(f"Try: {example}", key=f"example_{i}"):
                st.rerun()
    
    else:
        st.warning("Please initialize the RAG system using the sidebar.")

if __name__ == "__main__":
    main()