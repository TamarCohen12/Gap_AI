import os
import json
import hashlib
from typing import List
# import pandas as pd
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

# Configuration
FILES_DIR = "files"
VECTORSTORE_DIR = "vectorDB"
SUPPORTED_EXTENSIONS = ['.xlsx', '.csv', '.txt', '.json']

# Ensure directories exist
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

try:
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
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

def create_optimized_documents(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    for item in data:
        # יצירת טקסט מאוד ממוקד לחיפוש
        budgets = [
            f"{b['קוד_תקציב']} {b['שם_תקציב']}"
            for b in item.get('תקציבים_מהם_ניתן_לקנות_את_המענה', [])
        ]
        print(f"budgets: {budgets}")
        searchable_text = f"""
            {item['שם_מענה']}
            {item.get('קוד_מענה', '')}
            {' | '.join(budgets)}
        """
        print(f"searchable_text: {searchable_text}")
        
        doc = Document(
            page_content=searchable_text.strip(),
            metadata={
                "code_maane": item['קוד_מענה'],
                "name_maane": item['שם_מענה'],
                "budgetsOfMaane": budgets,
                "source": None#os.path.basename(file_path),
                # הוסף כאן את שאר השדות הרלוונטיים
                # **{k: v for k, v in item.items() if k not in ['קוד_מענה', 'שם_מענה', 'תקציבים_מהם_ניתן_לקנות_את_המענה']}
            }
        )
        print(f"doc: {doc}")
        documents.append(doc)
    
    return documents

def create_vectorstore(documents: List[Document]) -> FAISS:
    """Create FAISS vector store from documents"""
    if not documents:
        raise ValueError("No documents provided")
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100000,
        chunk_overlap=0,
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