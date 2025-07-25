import asyncio
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
    try:
        return jsonify({'history': chatbot.history})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def chat():
    return render_template('my-form.html')

if __name__ == "__main__":
    app.run()