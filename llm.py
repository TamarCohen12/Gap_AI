import os
import json
from datetime import datetime
# import pandas as pd
from pathlib import Path
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from graph_state import AgentState
from langchain_aws import ChatBedrock
from langchain_core.documents import Document
import re

load_dotenv()

try:
    model = ChatBedrock(
        model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
        model_kwargs={"temperature": 0},
        region_name=os.getenv("AWS_REGION", "us-east-1")
)
except Exception as e:
    print(f"Error initializing AWS Bedrock: {e}")
    model = None

def retrieve_documents(state: AgentState) -> AgentState:
    """Retrieve relevant documents from vectorstore with metadata filtering"""
    question = state["question"]
    vectorstore = state["vectorstore"]
    
    if not vectorstore:
        return {**state, "retrieved_docs": [], "sources": []}
    
    try:
        # Create retriever with metadata filter for "מוסד" population
        retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": 6,
                "filter": {"אוכלוסיה": "מוסד"}  # Filter for "מוסד" population only
            }
        )
        
        # Retrieve relevant documents (now filtered by metadata first)
        docs = retriever.invoke(question)
        
        # If no documents found with "מוסד" filter, fallback to general search
        if not docs:
            print("No documents found with 'מוסד' filter, falling back to general search")
            retriever_fallback = vectorstore.as_retriever(search_kwargs={"k": 6})
            docs = retriever_fallback.invoke(question)
        
        # Extract sources
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        
        # Log filtering results for debugging
        if docs:
            populations = [doc.metadata.get("אוכלוסיה", "Unknown") for doc in docs]
            print(f"Retrieved {len(docs)} documents with populations: {set(populations)}")
        
        return {
            **state,
            "retrieved_docs": docs,
            "sources": list(set(sources))  # Remove duplicates
        }
        
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        # If metadata filtering fails, fallback to regular search
        try:
            print("Metadata filtering failed, falling back to regular search")
            retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
            docs = retriever.invoke(question)
            sources = [doc.metadata.get("source", "Unknown") for doc in docs]
            
            return {
                **state,
                "retrieved_docs": docs,
                "sources": list(set(sources))
            }
        except Exception as fallback_error:
            print(f"Fallback search also failed: {fallback_error}")
            return {**state, "retrieved_docs": [], "sources": []}

def generate_answer(state: AgentState) -> AgentState:
    """Generate answer using retrieved documents"""
    question = state["question"]
    docs = state["retrieved_docs"]
    user_info = state["user_info"]

    # Create context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = ChatPromptTemplate.from_messages([("system",
        """אתה עוזר חכם המומחה למציאת מענים לפי שאלת המשתמש.
        **הנחיות:**
        - ענה בעברית בלבד
        - השתמש אך ורק במידע מהמסמכים המצורפים
        - אל תמציא מידע שלא קיים במסמכים
        - ענה קונקרטי לפי המידע שיש ברשותך, אל תתן הסבר או פירוט שלא קיים במידע
        - אסור להמליץ או להעדיף מענה אחד על פני השני!! אלא אך ורק למצוא את המענה המתאים ביותר לצורך המשתמש 
        - אם השאילתא לא ממקדת למענה מסוים, אלא מתאימה לרוב המענים, הסבר את זה למשתמש ואל תתן סתם כמה מענים ראשונים
        - אם יש כמה פריטים מתאימים - החזר כמה שיותר - ועד חמש פריטים

        **אם אין מידע מתאים:** 
        "לא מצאתי מענים מתאימים לשאלתך. אנא דייק את החיפוש."

        **אם יש מידע מתאים:**
        " תשובה כמו זו, אך כל פעם בניסוח קצת אחר שיהיה גיוון: מצאתי מענים מתאימים לשאלתך: [שמות המענים]"

        **פורמט תגובה (JSON בלבד):**
        {{
            "answer": "התשובה כאן",
            "maanim": "קודי המענה מופרדים בפסיקים"
        }}
        הקשר מהמסמכים:
        {context}
        מידע על תקציבי המשתמש:
        {user_info}
        מספיקה התאמה של תקציב אחד שקיים למשתמש ומשויך למענה, אין צורך בהתאמה של כמה תקציבים.
        """), ("human", "{question}")])

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

def process_user_query(state: AgentState) -> AgentState:
    """Process user query and generate a search query"""
    question = state["question"]
    # TODO: call llm to generate search query
    return {**state, "search_query": question} 

