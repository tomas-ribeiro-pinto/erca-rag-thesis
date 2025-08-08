import sqlite3

from api.model.chatbot import Chatbot

CHATBOT_API_DB_PATH = "./databases/chatbot_instances.db"

def initialise_app():
    initialise_sqlite_database()
    available_chatbots = load_chatbots()
    if not available_chatbots:
        print("No chatbots available. Please check the database.")
    else:
        print(f"Loaded {len(available_chatbots)} chatbots from the database.")
    return available_chatbots

def initialise_sqlite_database():
    con = sqlite3.connect(CHATBOT_API_DB_PATH)
    cur = con.cursor()
    cur.execute('''
            CREATE TABLE IF NOT EXISTS chatbot_instances (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            llm_model TEXT NOT NULL,
            temperature REAL NOT NULL,
            num_predict INTEGER NOT NULL,
            vector_db_path TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
    ''')

    res = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chatbot_instances'")
    if res.fetchone() is None:
        print("Table 'chatbot_instances' was not created successfully.")
        con.close()
        return
    else:
        print("Table 'chatbot_instances' is ready for use.")

    cur.execute('''
            CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NULL
            )
    ''')

    res = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    if res.fetchone() is None:
        print("Table 'users' was not created successfully.")
        con.close()
        return
    else:
        print("Table 'users' is ready for use.")

    cur.execute('''
        CREATE TABLE IF NOT EXISTS user_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            chatbot_id INTEGER NOT NULL,
            content TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id),
            FOREIGN KEY (chatbot_id) REFERENCES chatbot_instances(id)
        )
    ''')
    res = cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user_history'")
    if res.fetchone() is None:
        print("Table 'user_history' was not created successfully.")
        con.close()
        return
    else:
        print("Table 'user_history' is ready for use.")


    res = cur.execute("SELECT COUNT(*) FROM chatbot_instances")
    if res.fetchone()[0] == 0:
        print("No chatbot instances found, inserting default instances.")
        # Insert a default chatbot instance
        cur.execute('''
            INSERT INTO chatbot_instances (llm_model, temperature, num_predict, vector_db_path)
            VALUES (?, ?, ?, ?)
        ''', ("llama3.1", 0.8, 4096, "./databases/rag_milvus.db"))
        cur.execute('''
            INSERT INTO chatbot_instances (llm_model, temperature, num_predict, vector_db_path)
            VALUES (?, ?, ?, ?)
        ''', ("gemma3", 0.8, 4096, "./databases/rag_milvus.db"))
        cur.execute('''
            INSERT INTO users (username)
            VALUES (?)
        ''', ("guest_user",))
        cur.execute('''
            INSERT INTO users (username)
            VALUES (?)
        ''', ("tomas",))

    con.commit()
    con.close()

def load_chatbots():
    """Load available chatbots from the database"""
    con = sqlite3.connect(CHATBOT_API_DB_PATH)
    cur = con.cursor()
    res = cur.execute("SELECT * FROM chatbot_instances")
    available_chatbots = {}
    
    for row in res.fetchall():
        chatbot_id = f"{row[0]}"
        available_chatbots[chatbot_id] = Chatbot({
            "chatbot_id": chatbot_id,
            "llm_model": row[1],
            "temperature": row[2],
            "num_predict": row[3],
            "vector_db_path": row[4]
        }, chatbot_api_db_path=CHATBOT_API_DB_PATH)

    con.close()
    return available_chatbots