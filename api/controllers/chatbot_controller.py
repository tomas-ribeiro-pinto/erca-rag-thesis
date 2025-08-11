import sqlite3
from api.controllers.database_controller import DatabaseController

class ChatbotController():
    def create_chatbot_instances_table():
        DatabaseController.create_table_query("""
            CREATE TABLE IF NOT EXISTS chatbot_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            llm_model TEXT NOT NULL,
            temperature REAL NOT NULL,
            num_predict INTEGER NOT NULL,
            vector_db_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """, "chatbot_instances")
    
    def get_all_chatbot_instances():
        return DatabaseController.execute_query("SELECT * FROM chatbot_instances")

    def create_chatbot_instance(llm_model, temperature, num_predict, vector_db_path):
        query = """
            INSERT INTO chatbot_instances (llm_model, temperature, num_predict, vector_db_path)
            VALUES (?, ?, ?, ?)
        """
        params = (llm_model, temperature, num_predict, vector_db_path)
        return DatabaseController.execute_query(query, params)