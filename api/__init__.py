"""
================================================================================
RAG Chatbot API for Education - Thesis Project
--------------------------------------------------------------------------------
Author: Tomás Pinto
Date: August 2025
Description:
    This file loads the configuration settings and initializes the application.
    It sets up logging, initializes the SQLite database, and loads available chatbots.
================================================================================
"""

import os
from logging.config import dictConfig

# Set environment variable to disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from api.controllers.chatbot_controller import ChatbotController
from api.controllers.document_chunk_controller import DocumentChunkController
from api.controllers.user_controller import UserController
from api.models.chatbot import Chatbot
from api.settings import CHATBOT_API_DB_PATH, CHATBOT_GUIDELINES, LLM_TEMPERATURE, LLM_MAX_TOKENS
from flask import Flask
from flask import current_app as app

def initialise_app():
    dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        }},
        'handlers': {'wsgi': {
            'class': 'logging.StreamHandler',
            'stream': 'ext://flask.logging.wsgi_errors_stream',
            'formatter': 'default'
        }},
        'root': {
            'level': 'INFO',
            'handlers': ['wsgi']
        }
    })

    app = Flask(__name__)
    
    # Initialize database within application context
    with app.app_context():
        initialise_sqlite_database()
    
    return app

def initialise_sqlite_database():
    """Initialise the SQLite database and create necessary tables."""
    ChatbotController.create_chatbot_instances_table()
    UserController.create_users_table()
    UserController.create_user_history_table()
    DocumentChunkController.create_document_chunks_table()

    chatbot_instances = ChatbotController.get_all_chatbot_instances()
    if len(chatbot_instances) == 0:
        app.logger.warning("No chatbot instances found, inserting default instances.")
        ChatbotController.create_chatbot_instance(name="Image Processing chatbot", area_expertise="Image Processing", module_name="Introduction to Image Processing", llm_model="llama3.1", 
                                                  temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS, system_guidelines=CHATBOT_GUIDELINES, 
                                                  documents_path="./documents", vector_db_path="./databases/rag_milvus.db", use_ollama=1)
        ChatbotController.create_chatbot_instance(name="chatbot2", area_expertise="Image Processing", module_name="Introduction to Image Processing", llm_model="gemma3", 
                                                  temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS, system_guidelines=CHATBOT_GUIDELINES, 
                                                  documents_path="./documents", vector_db_path="./databases/rag_milvus.db", use_ollama=1)
        ChatbotController.create_chatbot_instance(name="chatbot3", area_expertise="Image Processing", module_name="Introduction to Image Processing", llm_model="meta-llama/llama-4-maverick:free", 
                                                  temperature=LLM_TEMPERATURE, max_tokens=LLM_MAX_TOKENS, system_guidelines=CHATBOT_GUIDELINES, 
                                                  documents_path="./documents", vector_db_path="./databases/rag_milvus.db")

    users = UserController.get_all_users()
    if len(users) == 0:
        UserController.create_user("guest_user", "guest_user@example.com")

def load_chatbots():
    """Load available chatbots from the database and start them"""
    chatbot_instances = ChatbotController.get_all_chatbot_instances()

    available_chatbots = {}

    for row in chatbot_instances:
        chatbot_id = f"{row["id"]}"
        available_chatbots[chatbot_id] = ChatbotController.run_chatbot_instance(
            id=chatbot_id,
            name=row["name"],
            area_expertise=row["area_expertise"],
            module_name=row["module_name"],
            system_guidelines=row["system_guidelines"],
            llm_model=row["llm_model"],
            temperature=row["temperature"],
            max_tokens=row["max_tokens"],
            documents_path=row["documents_path"],
            vector_db_path=row["vector_db_path"],
            use_ollama=row["use_ollama"]
        )

    return available_chatbots

def get_available_chatbots():
    """Get a list of available chatbots."""
    available_chatbots = load_chatbots()
    if not available_chatbots:
        app.logger.warning("No chatbots available. Please check the database.")
    else:
        app.logger.info(f"Loaded {len(available_chatbots)} chatbots from the database.")
    return available_chatbots