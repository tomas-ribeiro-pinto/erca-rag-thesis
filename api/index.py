"""
================================================================================
RAG Chatbot API for Education - Thesis Project
--------------------------------------------------------------------------------
Author: Tomás Pinto
Date: August 2025
Description:
    This file implements an API for interacting with multiple chatbot
    instances. It provides endpoints for sending prompts to chatbots, retrieving
    chat history, listing available chatbots, and rendering a chat interface.
    User management and chat history are handled via a SQLite database.
================================================================================
"""

import asyncio
import sqlite3
from flask import Response
import json

from api.controllers.chatbot_controller import ChatbotController
from api.controllers.user_controller import UserController
from flask import jsonify, request, render_template
from flask import current_app as app

from api import initialise_app, get_available_chatbots
from api.models.chatbot import Chatbot
from api.settings import CHATBOT_API_DB_PATH

# Initialize the app and load chatbots
app = initialise_app()

with app.app_context():
    available_chatbots = get_available_chatbots()

@app.route('/chatbot/<chatbot_id>', methods=['GET'])
def chat(chatbot_id):
    """
        Render the chat interface for a specific chatbot.
    """
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot_id:
        return jsonify({'error': 'Please provide all mandatory parameters.'}), 400
    if chatbot_id not in available_chatbots.keys():
        return jsonify({'error': 'Chatbot not found'}), 404

    user_email = request.args.get('user_email', None)  # Get user_email from query params
    return render_template('my-form.html', chatbot_id=chatbot_id, user_email=user_email)

@app.route('/api/chatbot/list', methods=['GET'])
def get_available_chatbots():
    chatbot_instances = [
        {"name": chatbot.name, "id": chatbot.chatbot_id}
        for chatbot in available_chatbots.values()
    ]
    return jsonify({'available_chatbots': chatbot_instances})

@app.route('/api/chatbot/create', methods=['POST'])
def create_chatbot():
    data = request.get_json()
    chatbot_name = data.get('name')
    system_prompt = data.get('system_prompt')
    documents_path = data.get('documents_path')
    llm_model = data.get('llm_model')

    if not chatbot_name or not system_prompt or not documents_path or not llm_model:
        return jsonify({'error': 'Please provide all mandatory parameters.'}), 400

    id, vector_db_path = ChatbotController.create_chatbot_instance(chatbot_name, llm_model, 0.6, 4096, documents_path, vector_db_path=None)

    new_chatbot = Chatbot({
            "name": chatbot_name,
            "chatbot_id": id,
            "llm_model": llm_model,
            "temperature": 0.6,
            "num_predict": 4096,
            "vector_db_path": vector_db_path
        }, chatbot_api_db_path=CHATBOT_API_DB_PATH)
    available_chatbots[new_chatbot.id] = new_chatbot

    return jsonify({'is_created': True, 'chatbot_id': id}), 201

@app.route('/api/chatbot/delete', methods=['POST', 'DELETE'])
def delete_chatbot():
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict() or request.args.to_dict()
    
    chatbot_id = data.get('chatbot_id')

    if not chatbot_id:
        return jsonify({'error': 'Please provide all mandatory parameters.'}), 400

    if chatbot_id not in available_chatbots.keys():
        return jsonify({'error': 'Chatbot not found.'}), 404

    ChatbotController.delete_chatbot_instance(chatbot_id)
    available_chatbots.pop(chatbot_id, None)
    return jsonify({'is_deleted': True}, 200)

@app.route('/api/chatbot/<chatbot_id>/prompt', methods=['POST'])
def stream_prompt_to_chatbot(chatbot_id):
    """
        Stream chatbot response as it's generated using Server-Sent Events.
    """
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot_id:
        return jsonify({'error': 'Please provide all mandatory parameters.'}), 400
    if chatbot_id not in available_chatbots.keys():
        return jsonify({'error': 'Chatbot not found'}), 404

    user = get_user_from_request(request, chatbot_id)
    user_id = user['id']

    try:
        data = request.get_json()
        user_prompt = data.get('prompt', '')
        
        if not user_prompt:
            return jsonify({'error': 'No prompt provided'}), 400

        def generate():
            try:
                # Create event loop for async operations
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Get the async generator from the chatbot
                    async def run_streaming():
                        stream_gen = await chatbot.invoke(user_id, user_prompt)
                        async for chunk in stream_gen:
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    
                    # Convert async generator to sync generator
                    async_gen = run_streaming()
                    
                    # Iterate through the async generator synchronously:
                    # - Gets the next chunk from the async generator
                    # - Uses run_until_complete to wait for each chunk synchronously
                    # - Yields each chunk to Flask's response stream
                    # - Breaks when the async generator is exhausted

                    while True:
                        try:
                            chunk = loop.run_until_complete(async_gen.__anext__())
                            yield chunk
                        except StopAsyncIteration:
                            break
                            
                finally:
                    loop.close()
                    
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Cache-Control'
            }
        )
    except Exception as e:
        return jsonify({'error': "Something went wrong:" + str(e)}), 500

@app.route('/api/chatbot/<chatbot_id>/history', methods=['POST'])
def get_chatbot_history(chatbot_id):
    """Retrieve chat history for a specific chatbot"""
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot:
        return jsonify({'error': 'Chatbot not found'}), 404
    
    user = get_user_from_request(request, chatbot_id)

    print(f"Retrieving history for user_email: {user['email']}, chatbot_id: {chatbot_id}")
    try:
        output = UserController.get_user_history(user['id'], chatbot_id)

        if len(output) == 0:
            UserController.register_user_with_chatbot(user['id'], chatbot_id, user['username'])

        return jsonify({'history': output})
    except Exception as e:
        return jsonify({'error': "Something went wrong:" + str(e)}), 500

def get_user_from_request(request, chatbot_id):
    data = request.get_json()
    user_email = data.get('user_email', None)
    user_name = data.get('user_name', "")

    if user_email is None or user_email == "" or user_email == "None":
        user = UserController.get_guest_user(chatbot_id)
    else:
        user = UserController.get_user_by_email(user_email)
        if user is None:
            UserController.create_user(username=user_name, user_email=user_email)
            user = UserController.get_user_by_email(user_email)
            UserController.register_user_with_chatbot(user['id'], chatbot_id, user_name)
    return user

if __name__ == "__main__":
    app.run()