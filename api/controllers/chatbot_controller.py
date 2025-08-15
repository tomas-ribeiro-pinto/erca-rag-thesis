import sqlite3

from flask import app
from api.controllers.database_controller import DatabaseController
import os

from api.models.chatbot import Chatbot
from api.settings import CHATBOT_API_DB_PATH
from chatbot.rag_retriever import RagRetriever

class ChatbotController():
    def create_chatbot_instances_table():
        DatabaseController.create_table_query("""
            CREATE TABLE IF NOT EXISTS chatbot_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            system_prompt TEXT NOT NULL,
            llm_model TEXT NOT NULL,
            temperature REAL NOT NULL,
            num_predict INTEGER NOT NULL,
            documents_path TEXT NOT NULL,
            vector_db_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """, "chatbot_instances")
    
    def get_all_chatbot_instances():
        return DatabaseController.execute_query("SELECT * FROM chatbot_instances")

    def create_chatbot_instance(name, llm_model, temperature, num_predict, system_prompt, documents_path, vector_db_path=None):
        if vector_db_path is None:
            query = """
                INSERT INTO chatbot_instances (name, llm_model, temperature, system_prompt, num_predict, documents_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            params = (name, llm_model, temperature, system_prompt, num_predict, documents_path)
            id = DatabaseController.execute_query(query, params)
            # Create vector database with proper .db extension
            vector_db_path = f"./databases/rag_milvus_{id}.db"
            _retriever = RagRetriever(vector_db_path=vector_db_path, documents_path=documents_path)
            _retriever.save_documents(documents_path)
            
            # Update the record with the vector_db_path
            update_query = "UPDATE chatbot_instances SET vector_db_path = ? WHERE id = ?"
            DatabaseController.execute_query(update_query, (vector_db_path, id))
            
            return id, vector_db_path
        else:
            query = """
                INSERT INTO chatbot_instances (name, llm_model, temperature, system_prompt, num_predict, documents_path, vector_db_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            params = (name, llm_model, temperature, system_prompt, num_predict, documents_path, vector_db_path)
            id = DatabaseController.execute_query(query, params)
        return id, None

    def delete_chatbot_instance(chatbot_id):
        # Delete vector_db_path
        # Fetch the vector_db_path for the chatbot instance
        result = DatabaseController.execute_query(
            "SELECT vector_db_path FROM chatbot_instances WHERE id = ?", (chatbot_id,)
        )
        if result:
            vector_db_path = result[0][0]
            if os.path.exists(vector_db_path) and vector_db_path != "rag_milvus.db":
                try:
                    os.remove(vector_db_path)
                except Exception as e:
                    app.logger.error(f"Error deleting vector_db_path file: {e}")

        # First delete related user history
        query1 = "DELETE FROM user_history WHERE chatbot_id = ?"
        params1 = (chatbot_id,)
        DatabaseController.execute_query(query1, params1)
        
        # Then delete the chatbot instance
        query2 = "DELETE FROM chatbot_instances WHERE id = ?"
        params2 = (chatbot_id,)
        DatabaseController.execute_query(query2, params2)