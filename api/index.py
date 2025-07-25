import asyncio
from flask import Flask, jsonify, request

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

if __name__ == "__main__":
    app.run()