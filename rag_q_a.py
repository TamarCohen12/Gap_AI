import os
import json
import hashlib
from typing import List, Dict,Sequence, TypedDict, Annotated
from datetime import datetime
# import pandas as pd
from pathlib import Path
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from graph_state import AgentState

load_dotenv()

# class AgentState(TypedDict):
#     messages: Annotated[Sequence[BaseMessage], add_messages]
#     question: str
#     search_query: str
#     vectorstore: object
#     retrieved_docs: List[Document]
#     answer: str
#     sources: List[str]
#     user_info: Dict[str, float]



# Configuration
FILES_DIR = "files"
VECTORSTORE_DIR = "vectorDB"
SUPPORTED_EXTENSIONS = ['.xlsx', '.csv', '.txt', '.json']

# Ensure directories exist
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Initialize AWS Bedrock models
try:
    model = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={"temperature": 0},
        region_name=os.getenv("AWS_REGION", "us-east-1")
)
    
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    print("AWS Bedrock models initialized successfully")
except Exception as e:
    print(f"Error initializing AWS Bedrock: {e}")
    model = None
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
        # if file_ext == '.xlsx':
        #     # Load Excel file
        #     df = pd.read_excel(file_path)
        #     # Convert DataFrame to text representation
        #     content = f"Data from {os.path.basename(file_path)}:\n\n"
        #     content += df.to_string(index=False)
            
        #     # Add column information
        #     content += f"\n\nColumns: {', '.join(df.columns.tolist())}"
        #     content += f"\nTotal rows: {len(df)}"
            
        #     # Create metadata
        #     metadata = {
        #         "source": os.path.basename(file_path),
        #         "type": "excel",
        #         "columns": df.columns.tolist(),
        #         "rows": len(df)
        #     }
            
        #     documents.append(Document(page_content=content, metadata=metadata))
            
        # elif file_ext == '.csv':
        #     # Load CSV file
        #     df = pd.read_csv(file_path)
        #     content = f"Data from {os.path.basename(file_path)}:\n\n"
        #     content += df.to_string(index=False)
            
        #     content += f"\n\nColumns: {', '.join(df.columns.tolist())}"
        #     content += f"\nTotal rows: {len(df)}"
            
        #     metadata = {
        #         "source": os.path.basename(file_path),
        #         "type": "csv",
        #         "columns": df.columns.tolist(),
        #         "rows": len(df)
        #     }
            
        #     documents.append(Document(page_content=content, metadata=metadata))
            
        # el
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
            
            content = f"JSON data from {os.path.basename(file_path)}:\n\n"
            content += json.dumps(data, indent=2, ensure_ascii=False)
            
            metadata = {
                "source": os.path.basename(file_path),
                "type": "json"
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
def process_user_query(state: AgentState) -> AgentState:
    """Process user query and generate a search query"""
    question = state["question"]
    user_info = state["user_info"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """אתה עוזר שמטרתו לנסח שאלת חיפוש מדויקת. 
        על סמך השאלה של המשתמש והמידע על תקציביו, נסח שאלת חיפוש ממוקדת 
        שתעזור למצוא את המידע הרלוונטי ביותר במסמכים.
        
        מידע על המשתמש:
        {user_info}
        
        שאלת המשתמש:
        {question}
        
        נסח שאלת חיפוש קצרה ומדויקת שתכלול את המידע הרלוונטי."""),
        ("human", "מה צריכה להיות שאלת החיפוש?")
    ])
    
    try:
        chain = prompt | model | StrOutputParser()
        search_query = chain.invoke({
            "user_info": json.dumps(user_info, ensure_ascii=False),
            "question": question
        })
        print(f"--------Generated search query: {search_query}------------")
        return {**state, "search_query": search_query}
        
    except Exception as e:
        print(f"Error processing user query: {e}")
        return {**state, "search_query": question}  # Use original question if processing fails

# Define LangGraph nodes
def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve relevant documents from vectorstore"""
    # question = state["question"]
    print("--------Retrieving documents------------")
    search_query = state["search_query"]
    vectorstore = state["vectorstore"]
    
    if not vectorstore:
        return {**state, "retrieved_docs": [], "sources": []}
    
    try:
        # Retrieve relevant documents
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(search_query)
        
        # Extract sources
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        
        return {
            **state,
            "retrieved_docs": docs,
            "sources": list(set(sources))  # Remove duplicates
        }
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return {**state, "retrieved_docs": [], "sources": []}

def generate_answer(state: AgentState) -> AgentState:
    """Generate answer using retrieved documents"""
    question = state["question"]
    docs = state["retrieved_docs"]
    user_info = state["user_info"]

    # Create context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Create prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """אתה עוזר חכם של חיפוש מענים לפי המידע השייך למשתמש, והקבצים שניתנו לך
הנחיות:
1. ענה בעברית בלבד
2. התבסס רק על המידע שניתן בהקשר
3. אם אין מידע רלוונטי, אמר זאת בבירור
4. תן תשובות מדויקות ומועילות
5. השתמש במספרים וכמויות כשהם רלוונטיים
6. אם המידע חלקי, ציין זאת

הקשר מהמסמכים:
{context}
מידע על המשתמש:
{user_info} """),
        ("human", "{question}")
    ])
    
    try:
        # Generate response
        chain = prompt | model | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "user_info": json.dumps(user_info, ensure_ascii=False),
            "question": question
        })
        
        return {**state, "answer": answer}
        
    except Exception as e:
        print(f"Error generating answer: {e}")
        return {
            **state, 
            "answer": "מצטער, אירעה שגיאה ביצירת התשובה. אנא נסה שוב."
        }
    
# def create_workflow():
#     workflow = StateGraph(AgentState)
#     workflow.add_node("retrieve", retrieve_documents)
#     workflow.add_node("generate", generate_answer)
#     workflow.set_entry_point("process_query")
#     workflow.add_node("process_query", process_user_query)
#     workflow.add_edge("process_query", "retrieve")
#     workflow.add_edge("retrieve", "generate")
#     workflow.add_edge("generate", END)
#     return workflow.compile()

# app_graph = create_workflow()
