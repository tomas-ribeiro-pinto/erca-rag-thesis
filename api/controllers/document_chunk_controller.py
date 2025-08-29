from api.controllers.database_controller import DatabaseController

class DocumentChunkController():
    def create_document_chunks_table():
        DatabaseController.create_table_query("""
            CREATE TABLE IF NOT EXISTS document_chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chatbot_id INTEGER NOT NULL,
            document_name TEXT NOT NULL,
            uuid TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (chatbot_id) REFERENCES chatbot_instances (id)
            )
        """, "document_chunks")

    def get_all_document_chunks_from_chatbot(chatbot_id):
        query = "SELECT * FROM document_chunks WHERE chatbot_id = ?"
        params = (chatbot_id,)
        return DatabaseController.execute_query(query, params)

    def get_document_chunks_uuids_by_document_name(chatbot_id, document_name):
        query = "SELECT uuid FROM document_chunks WHERE chatbot_id = ? AND document_name = ?"
        params = (chatbot_id, document_name)
        return DatabaseController.execute_query(query, params)

    def create_document_chunk(chatbot_id, document_name, uuid):
        query = """
            INSERT INTO document_chunks (chatbot_id, document_name, uuid)
            VALUES (?, ?, ?)
        """
        params = (chatbot_id, document_name, uuid)
        return DatabaseController.execute_query(query, params)

    def delete_document_chunks_by_document_name(document_name, chatbot_id):
        query = "DELETE FROM document_chunks WHERE document_name = ? and chatbot_id = ?"
        params = (document_name, chatbot_id)
        return DatabaseController.execute_query(query, params)