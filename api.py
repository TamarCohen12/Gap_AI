from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
from rag import create_vectorstore, get_file_hash, load_document, load_vectorstore, save_vectorstore,embeddings
from graph import app_graph
from llm import model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
FILES_DIR = "files"
SUPPORTED_EXTENSIONS = ['.xlsx', '.csv', '.txt', '.json']

# Global variables
current_vectorstore = None
current_file_hash = None
system_initialized = False

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    global system_initialized, current_vectorstore
    
    if not model or not embeddings:
        return jsonify({
            "status": "error on init moduls of AWS Bedrock",
            "initialized": False
        })
    
    if system_initialized and current_vectorstore:
        return jsonify({
            "status": "system ready for use",
            "initialized": True
        })
    else:
        return jsonify({
            "status": "system not initialized",
            "initialized": False
        })

@app.route('/api/initialize', methods=['GET'])#TODO:change to POST
def initialize_system():
    """Initialize the RAG system"""
    # TODO:move it to redis
    global current_vectorstore, current_file_hash, system_initialized
    
    try:
        # Look for data file in files directory
        data_file = None
        for ext in SUPPORTED_EXTENSIONS:
            potential_file = os.path.join(FILES_DIR, f"data{ext}")
            if os.path.exists(potential_file):
                data_file = potential_file
                break
        
        if not data_file:
            return jsonify({
                "success": False,
                "status": f"no file in{FILES_DIR}",
                "error": "file not found"
            }), 400
        
        # Get file hash
        file_hash = get_file_hash(data_file)
        
        # Check if we need to create new vectorstore
        if current_file_hash != file_hash:
            print("Creating new vectorstore...")
            
            # Try to load existing vectorstore
            vectorstore = load_vectorstore(file_hash)
            
            if vectorstore is None:
                # Create new vectorstore
                print("Loading and processing document...")
                documents = load_document(data_file)
                
                if not documents:
                    return jsonify({
                        "success": False,
                        "status": "Error loading file",
                        "error": "Could not load the file"
                    }), 400
                
                print("Creating vectorstore...")
                vectorstore = create_vectorstore(documents)
                
                # Save vectorstore
                save_vectorstore(vectorstore, file_hash)
            
            current_vectorstore = vectorstore
            current_file_hash = file_hash
        
        system_initialized = True
        
        return jsonify({
            "success": True,
            "status": "successfully initialized system",
            "file_processed": os.path.basename(data_file)
        })
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        return jsonify({
            "success": False,
            "status": "error initializing system",
            "error": str(e)
        }), 500

@app.route('/api/ask', methods=['GET'])#TOOD:change to POST
def ask_question():
    """Process question and return answer"""
    global current_vectorstore
    
    try:
        # data = request.get_json()
        # question = data.get('question', '').strip()
        question = request.args.get('question', '').strip()

        
        if not question:
            return jsonify({
                "error": "Question not found",
                "answer": "Please enter a valid question"
            }), 400
        
        if not current_vectorstore:
            return jsonify({
                "error": " System not initialized",
                "answer": "The system has not been initialized yet. Please wait for the system to initialize."
            }), 400
        # user_info = request.args.get('user_info', '').strip()
        # user_info = "סל מנהיגות חינוכית, סל חינוך חברתי - קהילתי והעשרה, סל אוכלוסיות במיקוד"
        user_info = ["סל תשתיות בית ספריות", "סל מנהיגות חינוכית", "סל חינוך חברתי - קהילתי והעשרה", "סל אוכלוסיות במיקוד"]
        # user_info = "סל מנהיגות חינוכית"
        # Create initial state
        initial_state = {
            "messages": [],
            "question": question,
            "vectorstore": current_vectorstore,
            "retrieved_docs": [],
            "answer": "",
            "search_query": "",
            "sources": [],
            "user_info": user_info
        }
        print(f"----------Processing question: {question}-----------")
        # Run the workflow
        result = app_graph.invoke(initial_state)
        
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "question": question,
            "search_query": result["search_query"]
        })
        
    except Exception as e:
        print(f"Error processing question: {e}")
        return jsonify({
            "error": "Error processing question",
            "answer": "Sorry, an error occurred while processing the question. Please try again."
        }), 500

if __name__ == '__main__':
    print("Starting RAG Backend...")
    app.run(debug=True, host='0.0.0.0', port=5000)