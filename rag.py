import os
import json
import hashlib
from typing import List
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

VECTORSTORE_DIR = "vectorDB"

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

try:
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        # cohere.embed-multilingual-v3
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    print("AWS Bedrock models initialized successfully")
except Exception as e:
    print(f"Error initializing AWS Bedrock: {e}")
    # model = None
    embeddings = None

def get_file_hash(file_path: str) -> str:
    """Generate hash for file to track changes"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error generating file hash: {e}")
        return ""

def load_document(file_path: str) -> List[Document]:
    """Load and process document based on file type"""
    documents = []
    file_ext = Path(file_path).suffix.lower()
    try:     
        if file_ext == '.txt':
            # Load text file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            metadata = {
                "source": os.path.basename(file_path),
                "type": "text"
            }
            
            documents.append(Document(page_content=content, metadata=metadata))
            
        elif file_ext == '.json':
            # Load JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            #TODO: get the population from the json file
            for i, item in enumerate(data):
                if i < 10:
                    population = "מוסד"
                elif i < 20:
                    population = "רשות"
                else:
                    population = "מחז"

                content = json.dumps(item, indent=2, ensure_ascii=False)

                metadata = {
                    "source": os.path.basename(file_path),
                    "type": "json",
                    "index": i,
                    "אוכלוסיה": population
                }

                documents.append(Document(page_content=content, metadata=metadata))
            
    except Exception as e:
        print(f"Error loading document {file_path}: {e}")
        
    return documents

def create_vectorstore(documents: List[Document]) -> FAISS:
    """Create FAISS vector store from documents"""
    if not documents:
        raise ValueError("No documents provided")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")
    
    # Create vector store
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore

def save_vectorstore(vectorstore: FAISS, file_hash: str):
    """Save vectorstore to disk"""
    try:
        vectorstore_path = os.path.join(VECTORSTORE_DIR, f"vectorstore_{file_hash}")
        vectorstore.save_local(vectorstore_path)
        print(f"Vectorstore saved: {vectorstore_path}")
        return True
    except Exception as e:
        print(f"Error saving vectorstore: {e}")
        return False

def load_vectorstore(file_hash: str) -> FAISS:
    """Load existing vectorstore from disk"""
    try:
        vectorstore_path = os.path.join(VECTORSTORE_DIR, f"vectorstore_{file_hash}")
        if os.path.exists(vectorstore_path):
            vectorstore = FAISS.load_local(
                vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded existing vectorstore: {vectorstore_path}")
            return vectorstore
    except Exception as e:
        print(f"Error loading vectorstore: {e}")
    
    return None