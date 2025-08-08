import asyncio
import sqlite3
from api import initialise_app

from flask import Flask, jsonify, request, render_template

from api.model.user_prompt import UserPrompt, UserPromptSchema
from api.model.chatbot_output import ChatbotOutput, ChatbotOutputSchema
from api.model.chatbot import Chatbot

CHATBOT_API_DB_PATH = "./databases/chatbot_instances.db"

app = Flask(__name__)

# Initialize the app and load chatbots
global available_chatbots
available_chatbots = initialise_app()

@app.route('/chatbot/<chatbot_id>/prompt', methods=['POST'])
def chatbot_prompt(chatbot_id):
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot:
        return jsonify({'error': 'Chatbot not found'}), 404
    user_id = request.args.get('user_id', '1')  # Default to 'guest_user ID'
    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        
        response = asyncio.run(chatbot.get_response(user_id, user_prompt))
        
        return jsonify({'response': response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/<chatbot_id>/history', methods=['GET'])
def chatbot_history(chatbot_id):
    """Retrieve chat history for a specific chatbot"""
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot:
        return jsonify({'error': 'Chatbot not found'}), 404
    
    user_id = request.args.get('user_id', '1')  # Default to 'guest_user ID'
    print(f"Retrieving history for user_id: {user_id}, chatbot_id: {chatbot_id}")
    con = sqlite3.connect(CHATBOT_API_DB_PATH)
    cur = con.cursor()
    try:
        res = cur.execute('''
            SELECT * FROM user_history WHERE user_id = ? AND chatbot_id = ?
        ''', (user_id, chatbot_id))

        output = [dict((cur.description[i][0], value) \
            for i, value in enumerate(row)) for row in cur.fetchall()]

        return jsonify({'history': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        con.close()

@app.route('/', methods=['GET'])
def get_available_chatbots():
    return jsonify({'available_chatbots': list(available_chatbots.keys())})

@app.route('/chatbot/<chatbot_id>', methods=['GET'])
def chat(chatbot_id):
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot:
        return jsonify({'error': 'Chatbot not found'}), 404
    
    user_id = request.args.get('user_id', '1')  # Get user_id from query params
    return render_template('my-form.html', chatbot_id=chatbot_id, user_id=user_id)

if __name__ == "__main__":
    app.run()