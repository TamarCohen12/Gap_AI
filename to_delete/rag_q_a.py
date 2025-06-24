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

# Configuration
FILES_DIR = "files"
VECTORSTORE_DIR = "vectorDB"
SUPPORTED_EXTENSIONS = ['.xlsx', '.csv', '.txt', '.json']

# Ensure directories exist
os.makedirs(FILES_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# Initialize AWS Bedrock models
try:
#     model = ChatBedrock(
#         model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
#         model_kwargs={"temperature": 0},
#         region_name=os.getenv("AWS_REGION", "us-east-1")
# )
    
    embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        region_name=os.getenv("AWS_REGION", "us-east-1")
    )
    print("AWS Bedrock models initialized successfully")
except Exception as e:
    print(f"Error initializing AWS Bedrock: {e}")
    # model = None
    embeddings = None
# graph
# retrieve_documents, generate_answer,process_user_query

# api
# create_vectorstore, get_file_hash, create_optimized_documents, load_vectorstore, save_vectorstore,model,embeddings
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
# def process_user_query(state: AgentState) -> AgentState:
#     """Process user query and generate a search query"""
#     question = state["question"]
#     user_info = state["user_info"]
#     # context = f"המשתמש בעל התקציבים: {user_info} שאל: '{question}'"

#     # prompt = ChatPromptTemplate.from_messages([
#     #     ("system", """אתה מומחה ביצירת שאילתות חיפוש מדויקות למערכת RAG (Retrieval-Augmented Generation).

#     #     מבנה הנתונים:
#     #     - הנתונים מאורגנים כאוסף של מענים (maane).
#     #     - כל מענה כולל: קוד (maanecode), שם (maanename), ותקציבים (budgets).
#     #     - התקציבים מזוהים על ידי קוד (budgetcode) ושם (badgetName).
#     #     - המענים משויכים לתקציב-על (masterBudjet).

#     #     משימתך:
#     #     יצירת שאילתת חיפוש מדויקת שתמצא את המידע הרלוונטי ביותר במסמכים, בהתבסס על שאלת המשתמש והתקציבים הזמינים לו.

#     #     הנחיות:
#     #     1. התמקד במילות מפתח מהשאלה ומהתקציבים של המשתמש.
#     #     2. כלול מונחים ספציפיים כגון קודי תקציב או שמות מענים, אם רלוונטיים.
#     #     3. השתמש באופרטורים לוגיים (AND, OR) לשילוב מדויק של תנאים.
#     #     4. הוסף מונחים נרדפים או קשורים להרחבת החיפוש, אך שמור על רלוונטיות.
#     #     5. התחשב בהקשר הכללי של השאלה ובמטרת החיפוש.

#     #     מידע נתון:
#     #     שאלת המשתמש: {question}
#     #     תקציבי המשתמש: {user_info}

#     #     נא לייצר שאילתת חיפוש בפורמט הבא:
#     #     {{
#     #         "text_query": "שאילתת החיפוש כאן"
#     #     }}

#     #     חשוב: החזר את התשובה אך ורק בפורמט JSON המבוקש, ללא הסברים או טקסט נוסף.
#     #             """),
#     #     ("human", "מה צריכה להיות שאילתת החיפוש?")
#     # ])
    
#     # try:
#     #     chain = prompt | model | StrOutputParser()
#     #     search_query = chain.invoke({"question": question, "user_info": user_info})
#     #     print(f"--------Generated search query: {search_query}------------")
#     #     json_search_query = json.loads(search_query)
#     #     print(f"--------Parsed search query: {json_search_query['text_query']}------------")
#     # return {**state, "search_query": json_search_query["text_query"]}
#     # return {**state, "search_query": f"{question} ,!!מענים חפש מענים המתאימים לשאלה,החזר רק את המענים שמשויכים לתקציבים אלו בלבד: {user_info} !!אל תחזיר מענים שלא משויכים לתקציבים האלו"} 
#     # TODO: call llm to generate search query
#     return {**state, "search_query": question}  # Use original question if processing fails
        
#     # except Exception as e:
#     #     print(f"Error processing user query: {e}")
#     #     return {**state, "search_query": question}  # Use original question if processing fails

from collections import Counter

# def extract_keywords_from_query(query: str) -> List[str]:
#     """חילוץ מילות מפתח מהשאילתה של המשתמש"""
#     # נקה את השאילתה
#     cleaned_query = re.sub(r'[^\w\s]', ' ', query)
#     words = cleaned_query.split()
    
#     # הסר מילות עזר נפוצות בעברית
#     stop_words = {"של", "את", "עם", "על", "אל", "מה", "איך", "כיצד", "האם", "זה", "זו", "זאת", "מענה","קשור","שייך","לשאלה","למשתמש","מענים","תקציבים","תקציב","מענים","חיפוש","רלוונטי","רלוונטיות","מילת מפתח", "מילות מפתח", "מילות עזר", "מילת עזר", "חיפוש"}
    
#     # השאר רק מילים חשובות (יותר מ-2 אותיות ולא מילות עזר)
#     keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
#     return keywords

# def hybrid_search(vectorstore: FAISS, query: str, budgets: str, k: int = 10) -> List[Document]:
#     # TODO: change it to be smart and relevant
#     # TODO: try to change the JSON to TXT...
#     # חלץ מילות מפתח מהשאילתה
#     user_keywords = extract_keywords_from_query(query)
#     print(f"מילות מפתח שחולצו: {user_keywords}")
#     # חפש עם השאילתה המקורית
#     vector_results = vectorstore.similarity_search_with_score(query, k=k*4)
#     sorted_results = sorted(vector_results, key=lambda x: x[1])
#     top_k_results = sorted_results[:k]
#     print(f"מספר תוצאות חיפוש: {len(vector_results)}")
#     print(f"results: {top_k_results}")
#     return [doc for doc, score in top_k_results[:k]]
   
#     # סנן לפי רלוונטיות לשאילתה
#     filtered_results = []
    
#     for doc, score in vector_results:
#         content = doc.page_content.lower()
#         query_lower = query.lower()
        
#         # חשב ציון רלוונטיות
#         relevance_score = calculate_relevance(content, user_keywords, query_lower)
        
#         # קבל רק תוצאות עם רלוונטיות מינימלית
#         if relevance_score > 0.3:  # סף שאפשר להתאים
#             combined_score = (score * 0.6) + (relevance_score * 0.4)
#             filtered_results.append((doc, combined_score))
    
#     # מיין לפי הציון המשולב
#     filtered_results.sort(key=lambda x: x[1], reverse=True)
    
#     return [doc for doc, score in filtered_results[:k]]

# def calculate_relevance(content: str, keywords: List[str], query: str) -> float:
#     """חישוב ציון רלוונטיות בין התוכן לשאילתה"""
#     relevance_score = 0.0
    
#     # בדוק התאמה מלאה לשאילתה
#     if query in content:
#         relevance_score += 1.0
    
#     # בדוק כמה מילות מפתח מופיעות
#     keyword_matches = sum(1 for keyword in keywords if keyword in content)
#     if keywords:
#         keyword_ratio = keyword_matches / len(keywords)
#         relevance_score += keyword_ratio * 0.8
    
#     # בדוק מילים דומות/חלקיות
#     partial_matches = 0
#     for keyword in keywords:
#         if len(keyword) > 3:  # רק למילים ארוכות
#             if any(keyword[:3] in word for word in content.split()):
#                 partial_matches += 1
    
#     if keywords:
#         partial_ratio = partial_matches / len(keywords)
#         relevance_score += partial_ratio * 0.3
    
#     return min(relevance_score, 1.0)

# def retrieve_documents(state: AgentState) -> AgentState:
#     search_query = state["search_query"]
#     vectorstore = state["vectorstore"]
#     user_info = state["user_info"]
    
#     if not vectorstore:
#         return {**state, "retrieved_docs": [], "sources": []}
    
#     try:
#         # Assume the first budget in user_info is the budgets we want to filter by
#         budgets = user_info
#         docs = hybrid_search(vectorstore, search_query, budgets)
        
#         sources = [doc.metadata.get("source", "Unknown") for doc in docs]
#         return {
#             **state,
#             "retrieved_docs": docs,
#             "sources": list(set(sources))
#         }
#     except Exception as e:
#         print(f"Error retrieving documents: {e}")
#         return {**state, "retrieved_docs": [], "sources": []}

# def generate_answer(state: AgentState) -> AgentState:
#     """Generate answer using retrieved documents"""
#     question = state["question"]
#     docs = state["retrieved_docs"]
#     user_info = state["user_info"]

#     # Create context from retrieved documents
#     context = "\n\n".join([doc.page_content for doc in docs])
   

#     # Create prompt
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """
#         אתה עוזר חכם המומחה למציאת "מוצרים לרכישה המכונים 'מענים' (maane) , 
#          עליך למצוא מענים מתאימים לשאלת המשתמש, חפש התאמה אינואטיבית
#          אל תהיה רגיש לשגיאות כתיב, 
#          חשוב, לכל משתמש משויכים תקציבים מסוימים, וכל מענה משויך לתקציבים אחרים, יש להחזיר אך ורק מענים שמשויכים לתקציב שקיים למשתמש.
#          אם לא חוזרים נתונים מתאימים , אל תחזיר כלום, ענה למשתמש בצורה יפה שלא מצאת מענה מתאים, שינסה לחפש בצורה אחרת.
#          חשוב ביותר!!! אם אין הקשר מהמסמכים, אל תחזיר שום מידע.
#          אין להמציא או להביא מידע שלא קיים בברור במסמכים הרלוונטיים!!!!
#          יש להחזיר רק מידע שקיים במסמכים, משויך לתקציבי המשתמש, וקשור לשאלה.
#          עליך להיות אדיב ונחמד, ועם זאת תמציתי ביותר.
#          במידה וחוזר מידע רלוונטי - ענה בקצרה: מצאתי מענים מתאימים לשאלתך, ואת שמות המענים .
#          אין צורך לציין תקציבים!!
#                החזר את התגובה במבנה הJSON הבא:
#             {{
#                 "answer": "תשובה כאן",
#                 "maanim":"קודי המענה משורשרים בפסיקים"
#             }}
#         תחזיר את המבנה הזה בלבד ללא הסברים נוספים.
#           הקשר מהמסמכים:
#             {context}
#          אם אין ערכים בהקשר מהמסמכים - אל תחזיר שום מידע.
#          אם אתה לא מוצא ערכים תואמים בהקשר מהמסמכים - אל תחזיר שום מידע.
#          מאוד חשוב להשתמש רק במידע  מהמסמכים, אין להביא מידע משום מקור אחר!!
#         מידע על תקציבי המשתמש:
#             {user_info} """),
#                     ("human", "{question}")
#         ])
    
#     try:
#         # Generate response
#         chain = prompt | model | StrOutputParser()
#         answer = chain.invoke({
#             "context": context,
#             "user_info": json.dumps(user_info, ensure_ascii=False),
#             "question": question
#         })
        
#         return {**state, "answer": answer}
        
#     except Exception as e:
#         print(f"Error generating answer: {e}")
#         return {
#             **state, 
#             "answer": "מצטער, אירעה שגיאה ביצירת התשובה. אנא נסה שוב."
#         }
    