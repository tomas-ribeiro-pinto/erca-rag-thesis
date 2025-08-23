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

from logging.config import dictConfig

from api.controllers.chatbot_controller import ChatbotController
from api.controllers.document_controller import DocumentChunkController
from api.controllers.user_controller import UserController
from api.models.chatbot import Chatbot
from api.settings import CHATBOT_API_DB_PATH, CHATBOT_PERSONA
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
        ChatbotController.create_chatbot_instance(name="Image Processing chatbot", llm_model="llama3.1", 
                                                  temperature=0.6, num_predict=4096, system_persona=CHATBOT_PERSONA, 
                                                  documents_path="./documents", vector_db_path="./databases/rag_milvus.db")
        ChatbotController.create_chatbot_instance(name="chatbot2", llm_model="gemma3", 
                                                  temperature=0.6, num_predict=4096, system_persona=CHATBOT_PERSONA, 
                                                  documents_path="./documents", vector_db_path="./databases/rag_milvus.db")
        ChatbotController.create_chatbot_instance(name="chatbot3", llm_model="deepseek-r1", 
                                                  temperature=0.6, num_predict=4096, system_persona=CHATBOT_PERSONA, 
                                                  documents_path="./documents", vector_db_path="./databases/rag_milvus.db")

    users = UserController.get_all_users()
    if len(users) == 0:
        UserController.create_user("guest_user", "guest_user@example.com")

def load_chatbots():
    """Load available chatbots from the database and start them"""
    chatbot_instances = ChatbotController.get_all_chatbot_instances()

    available_chatbots = {}

    for row in chatbot_instances:
        chatbot_id = f"{row[0]}"
        available_chatbots[chatbot_id] = Chatbot({
            "name": row[1],
            "chatbot_id": chatbot_id,
            "system_persona": row[2],
            "llm_model": row[3],
            "temperature": row[4],
            "num_predict": row[5],
            "documents_path": row[6],
            "vector_db_path": row[7],
        }, chatbot_api_db_path=CHATBOT_API_DB_PATH)

    return available_chatbots

def get_available_chatbots():
    """Get a list of available chatbots."""
    available_chatbots = load_chatbots()
    if not available_chatbots:
        app.logger.warning("No chatbots available. Please check the database.")
    else:
        app.logger.info(f"Loaded {len(available_chatbots)} chatbots from the database.")
    return available_chatbots