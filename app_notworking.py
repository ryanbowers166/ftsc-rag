from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import logging
from typing import List, Tuple, Optional
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import aiplatform
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for RAG system
rag_model = None
initialized = False

def initialize_rag_system():
    """Initialize the RAG system with current Vertex AI API"""
    global rag_model, initialized
    
    PROJECT_ID = "ftscrag"
    LOCATION = "us-central1"
    
    try:
        logger.info("Initializing Vertex AI...")
        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        
        # Create model with Vertex AI Search integration
        # Note: This assumes you have a Vertex AI Search datastore set up
        # You'll need to replace 'YOUR_DATASTORE_ID' with your actual datastore ID
        rag_model = GenerativeModel(
            model_name="gemini-2.0-flash-001"
        )
        
        initialized = True
        logger.info("RAG system initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        return False

def search_documents(query: str, top_k: int = 7):
    """Search documents using Vertex AI Search"""
    try:
        # This is a placeholder for Vertex AI Search integration
        # You would typically use the Discovery Engine API here
        # For now, we'll simulate a search response
        
        # Example structure of what you'd get from Vertex AI Search
        search_results = {
            "results": [
                {
                    "document": {
                        "title": "Document Title",
                        "content": "Document content snippet..."
                    },
                    "relevance_score": 0.95
                }
            ]
        }
        return search_results
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        return None

# HTML template for the web interface (same as before)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Research Assistant</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2c3e50;
        }
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
            min-height: 100px;
            box-sizing: border-box;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
        }
        .btn:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        .loading {
            display: none;
            text-align: center;
            color: #3498db;
            margin: 20px 0;
        }
        .response {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            white-space: pre-wrap;
            line-height: 1.6;
        }
        .error {
            background: #e74c3c;
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .status {
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .status.success {
            background: #d5f4e6;
            border: 1px solid #27ae60;
            color: #27ae60;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>RAG Research Assistant</h1>
        
        <div id="status"></div>
        
        <form id="queryForm">
            <div class="form-group">
                <label for="query">Enter your research question:</label>
                <textarea id="query" name="query" placeholder="What papers discuss neural network optimization?"></textarea>
            </div>
            <button type="submit" class="btn" id="submitBtn">Search Papers</button>
        </form>
        
        <div class="loading" id="loading">
            <p>Analyzing papers and generating response...</p>
        </div>
        
        <div id="results"></div>
    </div>

    <script>
        function showStatus(message, type) {
            const statusDiv = document.getElementById('status');
            statusDiv.innerHTML = '<div class="status ' + type + '">' + message + '</div>';
        }
        
        function showError(message) {
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="error">' + message + '</div>';
        }
        
        // Check system status on load
        fetch('/status')
            .then(function(response) {
                return response.json();
            })
            .then(function(data) {
                if (data.initialized) {
                    showStatus('RAG system ready!', 'success');
                } else {
                    showStatus('RAG system not initialized', 'warning');
                }
            })
            .catch(function(error) {
                showStatus('Error checking system status', 'warning');
                console.error('Status check error:', error);
            });
        
        document.getElementById('queryForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            const query = document.getElementById('query').value;
            console.log('Query:', query);
            
            if (!query.trim()) {
                showError('Please enter a research question.');
                return;
            }
            
            // Show loading
            document.getElementById('loading').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('results').innerHTML = '';
            
            fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    query: query
                })
            })
            .then(function(response) {
                console.log('Response status:', response.status);
                return response.text();
            })
            .then(function(responseText) {
                console.log('Raw response:', responseText);
                
                let data;
                try {
                    data = JSON.parse(responseText);
                } catch (parseError) {
                    throw new Error('Invalid JSON response: ' + responseText.substring(0, 200));
                }
                
                console.log('Parsed data:', data);
                
                if (data.error) {
                    showError(data.error);
                } else if (data.response) {
                    const resultsDiv = document.getElementById('results');
                    
                    const title = document.createElement('h3');
                    title.textContent = 'Research Assistant Response';
                    
                    const responseDiv = document.createElement('div');
                    responseDiv.className = 'response';
                    responseDiv.textContent = data.response;
                    
                    resultsDiv.innerHTML = '';
                    resultsDiv.appendChild(title);
                    resultsDiv.appendChild(responseDiv);
                    
                    console.log('Response displayed successfully');
                } else {
                    showError('No response data received from server');
                }
            })
            .catch(function(error) {
                console.error('Fetch error:', error);
                showError('Error processing request: ' + error.message);
            })
            .finally(function() {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('submitBtn').disabled = false;
            });
        });
    </script>
</body>
</html> """

@app.route('/')
def index():
    # Read template from external file to avoid encoding issues
    try:
        with open('template.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback minimal template
        return """<!DOCTYPE html>
        <html><head><meta charset="UTF-8"><title>Error</title></head>
        <body><h1>Template file not found</h1>
        <p>Create template.html file in your project directory</p></body></html>"""

@app.route('/status')
def status():
    """Check if the RAG system is initialized"""
    return jsonify({'initialized': initialized})

@app.route('/query', methods=['POST'])
def query():
    """Process a research query"""
    global rag_model
    
    logger.info(f"Query endpoint called. Initialized: {initialized}")
    
    if not initialized or not rag_model:
        logger.error("RAG system not initialized")
        return jsonify({'error': 'RAG system not initialized'}), 500
    
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
        
        # Create a system prompt for research analysis
        system_prompt = """You are a research assistant analyzing technical conference papers. 
        Based on your knowledge and the query provided, identify and discuss relevant research papers 
        and methodologies that address the topic. Focus on:
        
        1. Direct technical connections to the query topic
        2. Related methodologies and approaches
        3. Key contributions and findings
        4. Recommendations for further reading
        
        Be precise about relevance and cite specific concepts when possible.
        
        Query: """
        
        full_query = system_prompt + user_query
        logger.info("Generating content with Gemini...")
        
        # Generate response using Gemini
        response = rag_model.generate_content(full_query)
        logger.info("Content generated successfully")
        
        result = {'response': response.text}
        logger.info(f"Returning result with {len(response.text)} characters")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error processing query: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint for Cloud Run"""
    return jsonify({'status': 'healthy', 'initialized': initialized})

@app.route('/test')
def test():
    """Simple test endpoint"""
    return jsonify({'message': 'Flask server is working!', 'initialized': initialized})

@app.route('/initialize', methods=['POST'])
def manual_initialize():
    """Manual initialization endpoint"""
    logger.info("Manual initialization requested")
    success = initialize_rag_system()
    return jsonify({
        'success': success,
        'initialized': initialized,
        'message': 'Initialization successful' if success else 'Initialization failed'
    })

if __name__ == '__main__':
    # Initialize RAG system on startup
    logger.info("Starting Flask app and initializing RAG system...")
    initialize_rag_system()
    
    # Run the app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)