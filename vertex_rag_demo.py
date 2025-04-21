# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Works when run in cloud editor
# https://shell.cloud.google.com/cloudshell/editor?show=ide

import os

from typing import List, Tuple

from vertexai import rag
from vertexai.generative_models import GenerationResponse

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")


def quickstart(
        display_name: str,
        paths: List[str],
) -> Tuple[rag.RagCorpus, GenerationResponse]:
    from vertexai import rag
    from vertexai.generative_models import GenerativeModel, Tool
    import vertexai

    # Create a RAG Corpus, Import Files, and Generate a response

    PROJECT_ID = "ftsc-rag-demo"
    display_name = "test_corpus"
    paths = [
        "https://drive.google.com/drive/folders/1WO1rXTSVre9fsiY1souLMBnqybOcTOJH"]  # Supports Google Cloud Storage and Google Drive Links

    # Initialize Vertex AI API once per session
    vertexai.init(project=PROJECT_ID, location="us-central1")

    # Create RagCorpus
    # Configure embedding model, for example "text-embedding-005".
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
        # Optional
        transformation_config=rag.TransformationConfig(
            chunking_config=rag.ChunkingConfig(
                chunk_size=512,
                chunk_overlap=100,
            ),
        ),
        max_embedding_requests_per_min=1000,  # Optional
    )

    # Direct context retrieval
    rag_retrieval_config = rag.RagRetrievalConfig(
        top_k=3,  # Optional
        filter=rag.Filter(vector_distance_threshold=0.5),  # Optional
    )
    response = rag.retrieval_query(
        rag_resources=[
            rag.RagResource(
                rag_corpus=rag_corpus.name,
                # Optional: supply IDs from `rag.list_files()`.
                # rag_file_ids=["rag-file-1", "rag-file-2", ...],
            )
        ],
        text="Which papers in the sources are relevant for flight test?",
        rag_retrieval_config=rag_retrieval_config,
    )
    print(response)

    # Enhance generation
    # Create a RAG retrieval tool
    rag_retrieval_tool = Tool.from_retrieval(
        retrieval=rag.Retrieval(
            source=rag.VertexRagStore(
                rag_resources=[
                    rag.RagResource(
                        rag_corpus=rag_corpus.name,  # Currently only 1 corpus is allowed.
                        # Optional: supply IDs from `rag.list_files()`.
                        # rag_file_ids=["rag-file-1", "rag-file-2", ...],
                    )
                ],
                rag_retrieval_config=rag_retrieval_config,
            ),
        )
    )

    # Create a Gemini model instance
    rag_model = GenerativeModel(model_name="gemini-2.0-flash-001", tools=[rag_retrieval_tool])

    # Generate response
    print("\nRAG Query Console. Type 'exit' to quit.")
    while True:
        query = input("\nEnter your query: ")
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

        # Generate response
        print("\nGenerating response...")
        response = rag_model.generate_content(query)
        print("\nResponse:")
        print(response.text)

    # response = rag_model.generate_content("I want to plan a flight test of an autonomous vehicle. Which of the sources are relevant for me?")
    # print(response.text)
    return rag_corpus, response


if __name__ == "__main__":
    gdrive_path = "https://drive.google.com/file/1234567890"
    gcloud_path = "gs://your-bucket-name/file.txt"
    quickstart(
        display_name="test_corpus",
        paths=[gdrive_path, gcloud_path],
    )