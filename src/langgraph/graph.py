"""
KB금융그룹 RAG Agent - Graph Architecture
========================================
Main workflow implementation with proper node organization, 
routing, and subgraph composition.
"""

import json
from typing import Dict, Any, Optional, List
from functools import partial

from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph

from .models import RAGState
from .nodes import (
    session_init_node,
    intent_classification_node,
    supervisor_node,
    supervisor_router,
    chitchat_node,
    general_faq_node,
    product_extraction_node,
    product_search_node,
    session_summary_node,
    rag_search_node,
    guardrail_check_node,
    answer_node
)
from ..slm.slm import SLM


# ========== Node Constants ==========

# Main workflow nodes
SESSION_INIT = "session_init"
INTENT_CLASSIFICATION = "intent_classification"
SUPERVISOR = "supervisor"
CHITCHAT = "chitchat"
GENERAL_FAQ = "general_faq"
PRODUCT_EXTRACTION = "product_extraction"
PRODUCT_SEARCH = "product_search"
SESSION_SUMMARY = "session_summary"
RAG_SEARCH = "rag_search"
GUARDRAIL_CHECK = "guardrail_check"
ANSWER = "answer"


# ========== Utility Functions ==========

def join_graph(response: dict) -> dict:
    """Join graph utility for subgraph integration"""
    messages = response.get("messages", [])
    if messages:
        return {"messages": [messages[-1]]}
    return response


# ========== Router Functions ==========

def start_router(state: RAGState) -> list[str]:
    """Initial routing - always go to session_init"""
    return [SESSION_INIT]


# ========== Main Graph Factory Function ==========

def create_rag_workflow(
    checkpointer: BaseCheckpointSaver | None = None,
    llm: Any = None
) -> Runnable:
    """
    Create RAG workflow following the specified scenario:
    SESSION_INIT -> INTENT_CLASSIFICATION -> SUPERVISOR -> SPECIALIZED_NODES -> ANSWER
    
    Args:
        checkpointer: Checkpoint saver for conversation persistence
        llm: Language model for supervisor node operations (required!)
        
    Returns:
        Compiled StateGraph workflow
    """
    # SLM 인스턴스 생성 (중복 제거)
    slm = SLM()
    if not llm:
        llm = slm.llm
    
    workflow = StateGraph(RAGState)
    
    # ========== Core Workflow Construction ==========
    
    # Initialize state
    workflow.add_node(SESSION_INIT, session_init_node)
    workflow.add_edge(START, SESSION_INIT)
    
    # Intent classification
    intent_classification_with_slm = partial(intent_classification_node, slm=slm)
    workflow.add_node(INTENT_CLASSIFICATION, intent_classification_with_slm)
    workflow.add_edge(SESSION_INIT, INTENT_CLASSIFICATION)
    
    # Direct to supervisor after intent classification
    workflow.add_edge(INTENT_CLASSIFICATION, SUPERVISOR)
    
    # Supervisor node - LLM-based routing only
    supervisor_with_llm = partial(supervisor_node, llm=llm, slm=slm)
    workflow.add_node(SUPERVISOR, supervisor_with_llm)
    
    workflow.add_conditional_edges(
        SUPERVISOR,
        supervisor_router,
        [CHITCHAT, GENERAL_FAQ, PRODUCT_EXTRACTION, PRODUCT_SEARCH, SESSION_SUMMARY, RAG_SEARCH, GUARDRAIL_CHECK, INTENT_CLASSIFICATION, ANSWER]
    )
    
    # Chitchat node for casual conversations
    chitchat_with_slm = partial(chitchat_node, slm=slm)
    workflow.add_node(CHITCHAT, chitchat_with_slm)
    workflow.add_edge(CHITCHAT, ANSWER)
    
    # General FAQ node for banking questions
    general_faq_with_slm = partial(general_faq_node, slm=slm)
    workflow.add_node(GENERAL_FAQ, general_faq_with_slm)
    workflow.add_edge(GENERAL_FAQ, ANSWER)
    
    # Product extraction node
    product_extraction_with_slm = partial(product_extraction_node, slm=slm)
    workflow.add_node(PRODUCT_EXTRACTION, product_extraction_with_slm)
    workflow.add_edge(PRODUCT_EXTRACTION, PRODUCT_SEARCH)
    
    # Product search node
    product_search_with_slm = partial(product_search_node, slm=slm)
    workflow.add_node(PRODUCT_SEARCH, product_search_with_slm)
    workflow.add_edge(PRODUCT_SEARCH, ANSWER)
    
    # Session summary node for first turn
    session_summary_with_slm = partial(session_summary_node, slm=slm)
    workflow.add_node(SESSION_SUMMARY, session_summary_with_slm)
    workflow.add_edge(SESSION_SUMMARY, RAG_SEARCH)
    
    # RAG search node (VectorStore 인스턴스 재사용)
    from ..rag.vector_store import VectorStore
    vector_store = VectorStore()
    vector_store.get_index_ready()  # 한 번만 초기화
    rag_search_with_slm = partial(rag_search_node, slm=slm, vector_store=vector_store)
    workflow.add_node(RAG_SEARCH, rag_search_with_slm)
    workflow.add_edge(RAG_SEARCH, ANSWER)
    
    # Guardrail check node
    guardrail_check_with_slm = partial(guardrail_check_node, slm=slm)
    workflow.add_node(GUARDRAIL_CHECK, guardrail_check_with_slm)
    workflow.add_edge(GUARDRAIL_CHECK, ANSWER)
    
    # Answer node - final response generation
    workflow.add_node(ANSWER, answer_node)
    workflow.add_edge(ANSWER, END)
    
    # Compile workflow (성능 최적화를 위해 디버그 모드 비활성화)
    return workflow.compile(
        checkpointer=checkpointer or MemorySaver(),
        debug=False
    )


# ========== Factory Functions ==========

def create_agent():
    """Create agent instance with automatic LLM configuration"""
    from .agent import RAGAgent
    return RAGAgent()


# ========== Simple Chat Interface Functions ==========
# Simplified interface without heavy dependencies

def chat_send_message(message: str, session_id: str = "default") -> Dict[str, Any]:
    """Send message in chat - simplified interface"""
    try:
        # Create workflow with default settings
        workflow = create_rag_workflow()
        
        # Basic state for testing
        initial_state = {
            "messages": [],
            "query": message,
            "category": "",
            "product_name": "",
            "retrieved_docs": [],
            "context_text": "",
            "response": "",
            "sources": [],
            "session_context": None,
            "conversation_history": [],
            "turn_id": "",
            "intent_category": "",
            "guardrail_decision": "",
            "violations": [],
            "compliant_response": "",
        }
        
        result = workflow.invoke(initial_state)
        return {
            "response": result.get("response", "처리 중 오류가 발생했습니다."),
            "session_id": session_id,
            "success": True
        }
    except Exception as e:
        return {
            "response": f"오류가 발생했습니다: {str(e)}",
            "session_id": session_id,
            "success": False,
            "error": str(e)
        }


def api_chat(message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """API method for web frontends - simplified interface"""
    return chat_send_message(message, session_id or "default")
