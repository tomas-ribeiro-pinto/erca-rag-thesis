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

from api.controllers.user_controller import UserController
from flask import jsonify, request, render_template
from flask import current_app as app

from api import initialise_app, get_available_chatbots
from api.models.chatbot import Chatbot

# Initialize the app and load chatbots
app = initialise_app()

with app.app_context():
    available_chatbots = get_available_chatbots()

@app.route('/', methods=['GET'])
def get_available_chatbots():
    return jsonify({'available_chatbots': list(available_chatbots.keys())})

@app.route('/chatbot/<chatbot_id>', methods=['GET'])
def chat(chatbot_id):
    """
        Render the chat interface for a specific chatbot.
    """
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot:
        return jsonify({'error': 'Chatbot not found'}), 404

    user_email = request.args.get('user_email', None)  # Get user_email from query params
    return render_template('my-form.html', chatbot_id=chatbot_id, user_email=user_email)

@app.route('/chatbot/<chatbot_id>/prompt', methods=['POST'])
def stream_prompt_to_chatbot(chatbot_id):
    """
        Stream chatbot response as it's generated using Server-Sent Events.
    """
    chatbot = available_chatbots.get(chatbot_id)
    if not chatbot:
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
                        stream_gen = await chatbot.get_response_as_stream(user_id, user_prompt)
                        async for chunk in stream_gen:
                            yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    
                    # Convert async generator to sync generator
                    async_gen = run_streaming()
                    
                    # Iterate through the async generator synchronously
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
        return jsonify({'error': str(e)}), 500

@app.route('/chatbot/<chatbot_id>/history', methods=['POST'])
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
        return jsonify({'error': str(e)}), 500

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