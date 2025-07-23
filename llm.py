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
                # "filter": {'code_maane': '1323' } 
            }
        )
        
        # Retrieve relevant documents (now filtered by metadata first)
        docs = retriever.invoke(question)
        
        # If no documents found with "מוסד" filter, fallback to general search
        if not docs:
            print("No documents found, falling back to general search")
            retriever_fallback = vectorstore.as_retriever(search_kwargs={"k": 6})
            docs = retriever_fallback.invoke(question)
        for doc in docs:
            print(doc.metadata.get("source"), type(doc.metadata.get("source")))
        
        # Extract sources
        sources = [doc.metadata.get("source", "Unknown") for doc in docs]
        
        # Log filtering results for debugging
        if docs:
            populations = [doc.metadata.get("population", "Unknown") for doc in docs]
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

def classify_message(state: AgentState) -> AgentState:
    """Classify the incoming message"""
    question = state["question"]
    prompt = ChatPromptTemplate.from_messages([
    ("system",
     """אתה עוזר חכם ומיומן, המתמחה בזיהוי שאלות המשתמש וסיווגן לפי מטרתן לצורך איתור מענים.
        השאלה היא: {question}

        סווג את השאלה לאחד מהסוגים הבאים:
        1. שאלה כללית – אינה מבקשת מענה או מידע קונקרטי, לדוגמה: "שלום", "מה שלומך?", "תודה", וכדומה.
        2. שאלה ממוקדת – עוסקת בחיפוש מענה, פתרון, מידע או תכנית מסוימת.

        אם יש ספק כלשהו – סווג כשאלה ממוקדת (סוג 2).

        במקרה של שאלה כללית:
        - השב בנימוס, אך החזר את המשתמש בעדינות ובאסרטיביות למוקד השיחה – מציאת מענים.
        - דוגמה לתגובה מתאימה: "תודה! איך אפשר לעזור לך במציאת מענה מסוים?"

        החזר תשובה במבנה JSON לפי הדוגמאות הבאות:

        שאלה כללית (סוג 1):
        {{ 
            "message_type": "general_msg",
            "answer":"תשובה מנומסת אך ממוקדת"
        }}

        שאלה על חיפוש (סוג 2):
        {{ 
            "message_type": "search_msg",
            "answer": ""
        }}
        """),
            ("human", "{question}")
        ])

    try:
        # Generate classification response
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({"question": question})
        print(f"Classification response: {response}")
        # Parse the response
        data = json.loads(response)
        if data:
            message_type = data["message_type"] or "search_msg"        
            answer = data["answer"] or ""     
            print(f"Parsed classification: {message_type}, answer: {answer}")
            # return {**state, "answer": "ששש"}
            return {**state,  "message_type": message_type,  "answer": answer}
        else:
            print("Failed to parse classification response")
            return {**state, "message_type": "search_msg", "answer": ""}
    except Exception as e:
        print(f"Error classifying message: {e}")
        return {**state, "message_type": "search_msg", "answer": ""}

def generate_answer(state: AgentState) -> AgentState:
    """Generate answer using retrieved documents"""
    question = state["question"]
    user_info = state["user_info"]
    print(f"Generating answer for question: {question}")
    chat_history = state.get("chat_history", [])
    print(f"Generating answer for chat_history: {chat_history}")

    # Create context from retrieved documents
    with open("files/short_long_maanim.json", 'r', encoding='utf-8') as f:
        context = json.load(f)
    
    # יצירת הקשר של היסטוריית צ'אט
    history_context = ""
    if chat_history:
        history_context = "היסטוריית השיחה:\n" + "\n".join([
            f"- {item}" for item in chat_history[-3:]  # רק 3 הרשומות האחרונות
        ]) + "\n\n"
    
    prompt = ChatPromptTemplate.from_messages([("system",
        """אתה עוזר חכם המומחה למציאת מענים לפי שאלת המשתמש.
        **הנחיות:**
        - ענה בעברית בלבד
        - השתמש אך ורק במידע מהמסמכים המצורפים
        - אל תמציא מידע שלא קיים במסמכים
        - כשמבקשים מענה - תחזיר רק מענים ולא תקציבים
        - ענה קונקרטי לפי המידע שיש ברשותך, אל תתן הסבר או פירוט שלא קיים במידע
        - לרוב המשתמש ישאל שאלות הנוגעות למענים, הבן כך את השאלה. לדוגמא: שאלה: משהו שקשור לחשבון - תחפש מענה הקשור לחשבון
        - אסור להמליץ או להעדיף מענה אחד על פני השני!! אלא אך ורק למצוא את המענה המתאים ביותר לצורך המשתמש 
        - אם יש כמה פריטים מתאימים - החזר כמה שיותר - ועד חמש פריטים
        - שים לב להיסטוריית השיחה ולהקשר של שאלות קודמות
        -חשוב מאוד לשמור על רצף השיחה, לדוג' אם בהסטורית השיחה הוחזרו עמה מענים על פי בקשת המשתמש, ובשאלה הנוכחית הוא מבקש עוד, או מינים אחרים, אם אם יש נוספים - יש להביא מענים העונים על השאלה מההסטוריה, אך שלא חזרו בתשובה מההסטוריה.
        לדוג': אם המשתמש שאל בהסטוריה על מענה מתקציב קבוע וקיבל מענה 123, ובשאלה הנוכחית הוא מבקשה אם יש עוד מענים - יש לחפש עוד מענים ששייכים לתקציב קבוע ושהם לא קוד 123.
        - אם המשתמש אומר "עוד" או "נוספים" - חזור למענים שלא הוזכרו בתשובות קודמות
        -אם המשתמש מבקש פירוט/הסבר/הבדלים/מידע נוסף על המענים שחזרו בתשובה קודמת, אין צורך לחפש שוב מענים, אלא רק לפרט או לתת את המידע על המענים שחזרו בתשובה קודמת. 

        - אם המשתמש מבקש מידע על מענה מסוים, יש לחפש את המענה הזה ולתאר אותו.
        - אם המשתמש מבקש מידע על מענה מסוים, אך הוא לא קיים, יש להחזיר תשובה מתאימה.
        -אם אין מענה הקשור לשאלה באופן ישיר, תסביר זאת למשתמש, ואל תחזיר מענה שקשור באופן עקיף
        
        אם יש מידע מתאים, החזר את התשובה שצורה נעימה ואדיבה, אך חשוב לתמצת את התשובה שלא תהיה ארוכה מדי, כמו"כ - יש לכלול את שמות המענים המוחזרים בתשובה.       
        חשוב לציין את שמות המענים בתשובה!
        אם חוזרים מענים - יש לציין בתשובה את השם שלהם במקטע ה "answer" ואת הקודים עם השם שלהם במקטע: "maanim"
        חשוב מאוד להחזיר תגובה ב JSON במבנה הבא, ללא כל תוספות או מילים נוספות!!
        **(JSON בלבד):**
        {{
            "answer": "התשובה כאן כולל את שמות המענים שנמצאו",
            "maanim": "קוד המענה-שם המענה ואם יש כמה - להביא אותם מופרדים בפסיקים"
        }}
        אל תחזיר עוד מלל מעבר לJSON הזה הקפד על מבנה JSON תקין, ללא תוספת תווים או מרכאות שעלולים לשבור
        תבדוק את המבנה שהחזרת - שמדובר בJSON תקין לגמרי. מתחיל ונגמר ב-{{}} ומיכל את הערכים בצורה שיהיה ניתן להמיר לJSON. 
        אם יש תווים מיוחדים שמפריעים ל-JSON - בבקשה תוריד אותם, שהמבנה יהיה תקין לגמרי
        הקשר מהמסמכים:
        {context}
        מידע על תקציבי המשתמש:
        {user_info}
        שים לב למבנה התקציבים: code,teur, taktzivKavua.
        המשמעות של שדה taktzivKavua היא שנשאר מהתקציב הזה כסף שמכונה תקציב קבוע. אם המשתמש מעדיף לנצל תקציב קבוע - יש להחזיר לו מענים שמשוכים לתקציבים שהשדה הזה מכיל true.
        מספיקה התאמה של תקציב אחד שקיים למשתמש ומשויך למענה, אין צורך בהתאמה של כמה תקציבים.
        חשוב מאוד להחזיר רק מענים שמשויכים לתקציב שקיים למשתמש.
        אם אין למענה תקציב שמשויך למשתמש - אל תחזיר אותו!
        אם בשאלת המשתמש יש התיחסות לתקציב מסוים, או סוג תקציב מסוים, חפש מענים שייכים דווקא לתקציב המשויך.
        לדוגמא: המשתמש מבקש לסיים את תקציב "הכלה והשתלבות" - חפש רק מענים שמשויכים לתקציב הזה.
        עוד  דוגמא: המשתמש מבקש לקנות מענים מתקציב קבוע - תבדוק במידע על של תקציבי המשתמש באלו תקציבים השדה taktzivKavua מכיל true ותחזיר מענים ששייכים לתקציבים אלו.
        """), ("human", "{question} שאלות ותשובות קודמות: {history_context} ")])

    try:
        # Generate response
        chain = prompt | model | StrOutputParser()
        answer = chain.invoke({
            "context": context,
            "user_info": json.dumps(user_info, ensure_ascii=False),
            "question": question,
            "history_context": history_context
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
    # TODO: call llm to generate search query to RAG
    print(f"Processing user query: {question}")
    return {**state, "search_query": question} 

def create_summary(state: AgentState) -> AgentState:
    """יצירת תמצית קצרה של השאלה והתשובה"""
    question = state["question"]
    answer = state["answer"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """יצור תמצית קצרה של השאלה והתשובה המצורפים. 
         התמצית צריכה להיות קצרה ומדויקת - עד 100 מילים.
         אם חזר מידע על מענים ספציפיים בשאלה או בתשובה - חשוב מאוד לכלול אותם בתמצית
         התוכן שיחזור יראה כך:
         "תמצית קצרה של השאלה: ... התשובה: ..."
         השאלה: {question}
         התשובה: {answer}
         """),
        ("human", "יצור תמצית")
    ])
    
    try:
        chain = prompt | model | StrOutputParser()
        summary = chain.invoke({
            "question": question,
            "answer": answer
        })
        print(f"Summary response: {summary}")
        # data = json.loads(response)
        # print(data)
        # print(f"Generated summary: {data.get('summary', '')}")
        # summary = data.get("summary", f"שאלה: {question[:50]}... תשובה: {answer[:50]}...")
        
        return {**state, "summary": summary}
        
    except Exception as e:
        print(f"Error creating summary: {e}")
        # תמצית פשוטה במקרה של שגיאה
        summary = f"שאלה: {question[:50]}... תשובה: {answer[:50]}..."
        return {**state, "summary": summary}

