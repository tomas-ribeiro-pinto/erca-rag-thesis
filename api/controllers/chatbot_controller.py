from glob import glob
from flask import app
from api.controllers.database_controller import DatabaseController
from api.controllers.document_chunk_controller import DocumentChunkController
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
            area_expertise TEXT NOT NULL,
            module_name TEXT NOT NULL,
            system_guidelines TEXT NOT NULL,
            llm_model TEXT NOT NULL,
            temperature REAL NOT NULL,
            max_tokens INTEGER NOT NULL,
            documents_path TEXT NOT NULL,
            vector_db_path TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """, "chatbot_instances")

    def run_chatbot_instance(id, name, area_expertise, module_name, system_guidelines, llm_model, max_tokens, documents_path, vector_db_path, temperature, chatbot_api_db_path=CHATBOT_API_DB_PATH):
        chatbot_instance = Chatbot({
            "id": id,
            "name": name,
            "area_expertise": area_expertise,
            "module_name": module_name,
            "system_guidelines": system_guidelines,
            "llm_model": llm_model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "documents_path": documents_path,
            "vector_db_path": vector_db_path
        }, chatbot_api_db_path=chatbot_api_db_path)
        return chatbot_instance
    
    def get_all_chatbot_instances():
        return DatabaseController.execute_query("SELECT * FROM chatbot_instances")
    
    def get_chatbot_instance_by_id(id):
        return DatabaseController.execute_query("SELECT * FROM chatbot_instances WHERE id = ?", (id,))[0]

    def create_chatbot_instance(name, area_expertise, module_name, llm_model, temperature, max_tokens, system_guidelines, documents_path, vector_db_path=None):
        if vector_db_path is None:
            query = """
                INSERT INTO chatbot_instances (name, area_expertise, module_name, llm_model, temperature, system_guidelines, max_tokens, documents_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (name, area_expertise, module_name, llm_model, temperature, system_guidelines, max_tokens, documents_path)
            id = DatabaseController.execute_query(query, params)
            # Create vector database with proper .db extension
            vector_db_path = f"./databases/rag_milvus_{id}.db"
            _retriever = RagRetriever(vector_db_path=vector_db_path, documents_path=documents_path)
            _retriever.save_documents(documents_path)
            
            # Update the record with the vector_db_path
            update_query = "UPDATE chatbot_instances SET vector_db_path = ? WHERE id = ?"
            DatabaseController.execute_query(update_query, (vector_db_path, id))
        
        else:
            query = """
                INSERT INTO chatbot_instances (name, area_expertise, module_name, llm_model, temperature, system_guidelines, max_tokens, documents_path, vector_db_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = (name, area_expertise, module_name, llm_model, temperature, system_guidelines, max_tokens, documents_path, vector_db_path)
            id = DatabaseController.execute_query(query, params)

        chatbot_instance = ChatbotController.run_chatbot_instance(
            id = id, 
            name = name, 
            area_expertise = area_expertise, 
            module_name = module_name, 
            system_guidelines = system_guidelines,
            llm_model = llm_model, 
            temperature = temperature,
            max_tokens = max_tokens, 
            documents_path = documents_path,
            vector_db_path=vector_db_path, 
        )
         
        return id, chatbot_instance

    def update_chatbot_instance_settings(id, name, area_expertise, module_name, llm_model, system_guidelines, max_tokens):
        # Update chatbot instance settings in the database
        query = """
            UPDATE chatbot_instances SET
            name = ?,
            area_expertise = ?,
            module_name = ?,
            llm_model = ?,
            system_guidelines = ?,
            max_tokens = ?,
            WHERE id = ?
        """
        params = (
            name,
            area_expertise,
            module_name,
            llm_model,
            system_guidelines,
            max_tokens,
            id
        )
        DatabaseController.execute_query(query, params)

        instance = ChatbotController.get_chatbot_instance_by_id(id)

        chatbot_instance = ChatbotController.run_chatbot_instance(
            id=instance['id'],
            name=instance['name'],
            area_expertise=instance['instance'],
            module_name=instance['module_name'],
            system_guidelines=instance['system_guidelines'],
            llm_model=instance['llm_model'],
            max_tokens=instance['max_tokens'],
            documents_path=instance['documents_path'],
            vector_db_path=instance['vector_db_path']
        )
        return chatbot_instance
    
    def update_chatbot_instance_memory(instance, deleted_documents, added_documents):
        documents_not_updated = []
        for document in deleted_documents:
            uuids = instance.retriever.delete_document(document)
            if not uuids or uuids == []:
                documents_not_updated.append(document)

        for document in added_documents:
            docs = glob(f"{instance['documents_path']}/{document}")()
            if not docs or docs == []:
                documents_not_updated.append(document)
            else:
                instance.retriever.save_documents(docs)

        return documents_not_updated

    def delete_chatbot_instance(chatbot_id):
        # Delete vector_db_path
        # Fetch the vector_db_path for the chatbot instance
        result = ChatbotController.get_chatbot_instance_by_id(chatbot_id)
        if result:
            vector_db_path = result['vector_db_path']
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

        # Lastly, delete related documents
        query3 = "DELETE FROM documents WHERE chatbot_id = ?"
        params3 = (chatbot_id,)
        DatabaseController.execute_query(query3, params3)