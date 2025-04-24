import os
from typing import List, Tuple
from vertexai import rag
from vertexai.generative_models import GenerationResponse

# Create a RAG corpus, import files, and generate a response
def quickstart(
        display_name: str,
        paths: List[str],
) -> Tuple[rag.RagCorpus, GenerationResponse]:

    from vertexai import rag
    from vertexai.generative_models import GenerativeModel, Tool
    import vertexai

    PROJECT_ID = "ftsc-rag-demo"
    display_name = "test_corpus"
    paths = ["https://drive.google.com/drive/folders/1WO1rXTSVre9fsiY1souLMBnqybOcTOJH"]  # Supports Google Cloud Storage and Google Drive Links
    
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

    # Parameters
    top_k = 7  # Number of relevant sources to retrieve
    vector_distance_threshold = 0.5 
    llm_model_name = "gemini-2.0-flash-001"


    # Initialize Vertex AI API
    vertexai.init(project=PROJECT_ID, location="us-central1")

    # Create RagCorpus
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

    # Import Files to the RagCorpus
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

    # Direct context retrieval
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=top_k,
        filter=rag.Filter(vector_distance_threshold=vector_distance_threshold),
    )

    # Create RAG retrieval tool
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
                rag_retrieval_config=rag_retrieval_config,
            ),
        )
    )

    # Create a model instance
    rag_model = GenerativeModel(model_name=llm_model_name, tools=[rag_retrieval_tool])

    # Generate response
    print("\nRAG Query Console. Type 'exit' to quit.")
    while True:
        user_input = input("\nEnter your query: ")
        query = system_prompt + user_input
        
        if query.lower() == 'exit':
            break

        show_chunks = input("Show retrieved chunks? (y/n): ").lower() == 'y'
        if show_chunks:
            retrieval_response = rag.retrieval_query(
                rag_resources=[rag.RagResource(rag_corpus=rag_corpus.name)],
                text=query,
                rag_retrieval_config=rag_retrieval_config,
            )
            print("\nRetrieved chunks:")
            print(retrieval_response)

        print("\nGenerating response...")
        response = rag_model.generate_content(query)
        print("\nResponse:")
        print(response.text)

    return rag_corpus, response


if __name__ == "__main__":
    gdrive_path = "https://drive.google.com/drive/folders/1WO1rXTSVre9fsiY1souLMBnqybOcTOJH"
    quickstart(
        display_name="test_corpus",
        paths=[gdrive_path],
    )
