import sqlite3
from api.settings import CHATBOT_DEFAULT_GREETING_MESSAGE
from  api.controllers.database_controller import DatabaseController

class UserController():
    def create_users_table():
        DatabaseController.create_table_query("""
            CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NULL
            )
        """, "users")

    def create_user_history_table():
        DatabaseController.create_table_query("""
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            chatbot_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (chatbot_id) REFERENCES chatbot_instances(id)
        )
        """, "user_history")

    def create_user(username, user_email):
        DatabaseController.execute_query('''
        INSERT INTO users (username, email) VALUES (?, ?)
        ''', (username, user_email,))

    def add_user_history_entry(user_id, chatbot_id, content, role):
        DatabaseController.execute_query('''
            INSERT INTO user_history (user_id, chatbot_id, content, role)
            VALUES (?, ?, ?, ?)
        ''', (user_id, chatbot_id, content, role))

    def get_user_history(user_id, chatbot_id):
        rows = DatabaseController.execute_query('''
            SELECT * FROM user_history WHERE user_id = ? AND chatbot_id = ?
        ''', (user_id, chatbot_id))
        return [dict(row) for row in rows]

    def register_user_with_chatbot(user_id, chatbot_id, user_name):
        content = CHATBOT_DEFAULT_GREETING_MESSAGE
        content = content.format(user_name=user_name)

        role = "assistant"
        UserController.add_user_history_entry(user_id, chatbot_id, content, role)

    def get_user_by_email(email):
        rows = DatabaseController.execute_query('''
            SELECT * FROM users WHERE email = ?
        ''', (email,))
        return rows[0] if rows else None

    def get_user_by_id(user_id):
        rows = DatabaseController.execute_query('''
            SELECT * FROM users WHERE id = ?
        ''', (user_id,))
        return rows[0] if rows else None

    def get_all_users():
        query = "SELECT * FROM users"
        return DatabaseController.execute_query(query)

    def get_guest_user(chatbot_id):
        guest_email = "guest_user@example.com"
        user = UserController.get_user_by_email(guest_email)
        if user is None:
            UserController.create_user(username="guest_user", user_email=guest_email)
            user = UserController.get_user_by_email(guest_email)
            UserController.register_user_with_chatbot(user['id'], chatbot_id, user['username'])
        return user
