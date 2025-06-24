
from langgraph.graph import StateGraph, END
from graph_state import AgentState
from llm import retrieve_documents, generate_answer,process_user_query


def create_workflow():
    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.add_node("process_query", process_user_query)
    workflow.set_entry_point("process_query")
    workflow.add_edge("process_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()

app_graph = create_workflow()