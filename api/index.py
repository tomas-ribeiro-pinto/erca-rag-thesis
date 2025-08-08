import asyncio
import sqlite3
from flask import Flask, jsonify, request, render_template

from api.model.user_prompt import UserPrompt, UserPromptSchema
from api.model.chatbot_output import ChatbotOutput, ChatbotOutputSchema
from api.model.chatbot import Chatbot

app = Flask(__name__)

chatbot = Chatbot()

@app.route('/chatbot/prompt', methods=['POST'])
def chatbot_prompt():
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        response = asyncio.run(chatbot.get_response(user_prompt))
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/chatbot/history', methods=['GET'])
def chatbot_history():
    con = sqlite3.connect(chatbot.history_db_path)
    cur = con.cursor()
    try:
        res = cur.execute('''
            SELECT * FROM user_history WHERE user_id = ? AND chatbot_id = ?
        ''', ("guest_user", "rag_chatbot"))

        output = [dict((cur.description[i][0], value) \
            for i, value in enumerate(row)) for row in cur.fetchall()]

        return jsonify({'history': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        con.close()

@app.route('/', methods=['GET'])
def chat():
    return render_template('my-form.html')

if __name__ == "__main__":
    app.run()