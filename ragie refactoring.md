Refactoring Plan: Google Cloud Vertex AI → Ragie

  Based on my analysis of your current code and Ragie's capabilities, here's a comprehensive plan to refactor your FTSC RAG
  application:

  Current Architecture Overview

  Your app currently uses:
  - Google Cloud Vertex AI for RAG corpus management
  - Vertex AI Vector Search for document indexing
  - Google Drive API for file access and download links
  - Gemini 2.5 Flash as the LLM
  - Flask for the web server
  - HTML/JavaScript frontend with conversation history

  ---
  Refactoring Plan

  Phase 1: Setup & Dependencies

  1. Install Ragie Python SDK
    - Add ragie to requirements.txt
    - Remove Google Cloud dependencies: vertexai, google-cloud-aiplatform
    - Keep google-api-python-client and google-oauth2 (for Google Drive file links)
  2. Get Ragie API credentials
    - Sign up for Ragie account
    - Obtain API key from Ragie dashboard
    - Store API key as environment variable

  Phase 2: Backend Refactoring

  Components to Remove:
  - RAGSystem.create_corpus_with_vector_search() - Ragie manages this
  - RAGSystem.find_existing_vector_search_resources() - Not needed
  - RAGSystem.setup_authentication() - Replace with Ragie auth
  - RAGSystem.find_existing_corpus() - Ragie handles corpus management
  - All Vertex AI initialization code

  Components to Keep:
  - GoogleDriveHelper class - Still needed for file download links
  - Flask routes and structure
  - Conversation history management
  - HTML template (no changes needed)

  Components to Modify:

  1. RAGSystem class initialization
    - Replace Vertex AI client with Ragie client
    - Simplify authentication (just API key)
    - Remove corpus creation logic (assume corpus exists in Ragie)
  2. query_with_sources() method
    - Replace rag_model.generate_content() with Ragie's retrieval + LLM call
    - Use Ragie's /retrievals endpoint for document search
    - Option A: Use Ragie's built-in LLM
    - Option B: Use Ragie for retrieval only, then call external LLM (Gemini, Claude, etc.)
  3. direct_retrieval_query() method
    - Replace with Ragie's retrieval-only endpoint
    - Map parameters (top_k, threshold) to Ragie's API

  Phase 3: Code Structure

  New RAGSystem class structure:
  class RAGSystem:
      def __init__(self):
          self.ragie_client = None
          self.initialized = False
          self.drive_helper = GoogleDriveHelper()
          self.api_key = os.getenv('RAGIE_API_KEY')

      def initialize(self):
          # Initialize Ragie client with API key
          # Much simpler than current initialization

      def query_with_sources(self, user_query, conversation_history):
          # 1. Call Ragie retrieval API
          # 2. Get relevant chunks with metadata
          # 3. Format prompt with system instructions + context + history
          # 4. Call LLM (could use Ragie's or external)
          # 5. Extract sources from response
          # 6. Return formatted result

      def direct_retrieval_query(self, query_text, top_k):
          # Call Ragie's retrieval endpoint directly

  Phase 4: API Integration Details

  Ragie API Calls Needed:
  1. Retrieval: POST https://api.ragie.ai/retrievals
    - Send query text
    - Specify top_k, filters
    - Receive ranked document chunks with metadata
  2. Optional - RAG Generation: If using Ragie's built-in LLM
    - Ragie can handle both retrieval + generation
    - Simpler but less control over LLM

  Recommended Approach:
  - Use Ragie for retrieval only (get document chunks)
  - Call external LLM API (OpenAI, Anthropic, or Google Gemini) for generation
  - This gives you more control over prompts and LLM behavior

  Phase 5: Configuration Changes

  Environment Variables:
  - Remove: GOOGLE_CLOUD_PROJECT, DEVSHELL_PROJECT_ID
  - Add: RAGIE_API_KEY
  - Keep: Google Drive credentials (for file links)

  Flask Routes (minimal changes):
  - /initialize - Simpler, just create Ragie client
  - /query - Same interface, different backend
  - /status - Check Ragie connection instead of Vertex AI
  - Keep all other routes unchanged

  ---
  Key Benefits of This Refactoring

  ✅ Simplified Code: ~300 lines removed (no corpus management, no Vector Search setup)✅ Faster Initialization: No need to
  create/find indexes and endpoints✅ Same User Experience: HTML interface unchanged✅ Maintained Features: Conversation history,
   source links, all work the same✅ Corpus Management: Upload documents via Ragie web UI (easier than code)

  ---
  Migration Steps

  1. Pre-migration: Upload your PDF corpus to Ragie via their web interface
  2. Code changes: Refactor app.py to use Ragie SDK
  3. Testing: Test retrieval quality and source matching
  4. Deployment: Update environment variables and deploy

  ---
  Potential Challenges

  ⚠️ Source extraction: Your current code extracts PDF filenames from LLM responses. Need to ensure Ragie's metadata includes
  filenames.⚠️ Google Drive links: Still need Google Drive API for download links (unchanged)⚠️ LLM choice: Decide if using
  Ragie's LLM or external (I recommend external for consistency)⚠️ Conversation history: Need to manually manage context window
  (Ragie might not have built-in conversation history)

  ---
  Estimated Effort

  - Setup & credential: 30 minutes
  - Code refactoring: 2-3 hours
  - Testing & adjustments: 1-2 hours
  - Total: ~4-6 hours