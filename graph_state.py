from typing import Annotated, Dict, List, Sequence, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    search_query: str
    vectorstore: object
    retrieved_docs: List[Document]
    answer: str
    sources: List[str]
    user_info:str
