"""
LangGraph Tools
==============
RAG 워크플로우에서 사용하는 도구들
"""

from langchain_core.tools import tool

@tool(parse_docstring=True)
def chitchat(thought: str, message: str):
    """
    Handle casual conversation, greetings, thanks, and general chat.
    
    Args:
        thought: Reasoning for using this tool (1-3 sentences)
        message: User's message content for chitchat response
    """

@tool(parse_docstring=True)
def general_faq(thought: str, question: str):
    """
    Answer general banking FAQ questions using SLM knowledge.
    
    Args:
        thought: Reasoning for using this tool (1-3 sentences)
        question: User's general banking question
    """

@tool(parse_docstring=True)
def rag_search(thought: str, query: str):
    """
    Search documents using RAG for specific information.
    
    Args:
        thought: Reasoning for using this tool (1-3 sentences)
        query: Search query for document retrieval
    """

@tool(parse_docstring=True)
def product_extraction(thought: str, query: str):
    """
    Extract product name from user query.
    
    Args:
        thought: Reasoning for using this tool (1-3 sentences)
        query: User query to extract product name from
    """

@tool(parse_docstring=True)
def product_search(thought: str, product_name: str, query: str):
    """
    Search for specific product information.
    
    Args:
        thought: Reasoning for using this tool (1-3 sentences)
        product_name: Extracted product name
        query: User's question about the product
    """

@tool(parse_docstring=True)
def session_summary(thought: str, query: str):
    """
    Generate session summary for first conversation turn.
    
    Args:
        thought: Reasoning for using this tool (1-3 sentences)
        query: User's first message to summarize
    """

@tool(parse_docstring=True)
def guardrail_check(thought: str, response: str):
    """
    Check response against guardrails for compliance.
    
    Args:
        thought: Reasoning for using this tool (1-3 sentences)
        response: Generated response to check
    """

@tool(parse_docstring=True)
def intent_classification(thought: str, query: str):
    """
    Classify user query into one of four categories: general_banking_FAQs, 
    industry_policies_and_regulations, company_rules, or company_products.
    
    Args:
        thought: Reasoning for classification (1-2 sentences)
        query: User query to classify
    """

@tool(parse_docstring=True)
def answer(thought: str):
    """
    Generate final answer when sufficient information is available.
    
    Args:
        thought: Extremely concise reason why ready to answer (3-5 words maximum)
    """
