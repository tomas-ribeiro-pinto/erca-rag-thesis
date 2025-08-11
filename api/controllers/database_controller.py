import sqlite3
import logging
from api.settings import CHATBOT_API_DB_PATH
from flask import current_app as app

class DatabaseController():
    def execute_query(query, params=()):
        con = sqlite3.connect(CHATBOT_API_DB_PATH)
        con.row_factory = sqlite3.Row  # This makes rows dict-like
        cur = con.cursor()
        res = cur.execute(query, params)
        data = res.fetchall()
        con.commit()
        con.close()
        return data

    def create_table_query(query, table_name, params=()):
        if not table_name:
            raise ValueError("Table name must be provided")

        DatabaseController.execute_query(query, params)
        res = DatabaseController.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        
        # Log the result
        if res is None:
            app.logger.error(f"Table '{table_name}' was not created successfully.")
        else:
            app.logger.info(f"Table '{table_name}' is ready for use.")
