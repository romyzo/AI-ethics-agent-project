"""
AI Ethics Agent - Utility Functions Package

This package contains common utility functions used across all agents.
"""

from .llm_utils import call_llm, call_llm_with_json_output, calculate_token_count
from .document_utils import load_document, chunk_text, extract_metadata
from .vector_store_utils import initialize_vector_store, search_similar_documents, add_documents_to_store
from .prompt_utils import load_prompt_template, format_prompt
from .validation_utils import validate_risk_assessment, calculate_confidence_score, sanitize_input
from .logging_utils import setup_logger, log_agent_activity, log_performance_metrics
from .web_search_utils import search_web, scrape_webpage, search_news

__all__ = [
    # LLM utilities
    'call_llm',
    'call_llm_with_json_output',
    'calculate_token_count',
    
    # Document utilities
    'load_document',
    'chunk_text',
    'extract_metadata',
    
    # Vector store utilities
    'initialize_vector_store',
    'search_similar_documents',
    'add_documents_to_store',
    
    # Prompt utilities
    'load_prompt_template',
    'format_prompt',
    
    # Validation utilities
    'validate_risk_assessment',
    'calculate_confidence_score',
    'sanitize_input',
    
    # Logging utilities
    'setup_logger',
    'log_agent_activity',
    'log_performance_metrics',
    
    # Web search utilities
    'search_web',
    'scrape_webpage',
    'search_news',
]

