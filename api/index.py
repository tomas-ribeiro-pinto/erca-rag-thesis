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
from api.settings import LLM_TEMPERATURE

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
    isRequired = True

    fields = [
        ('name', isRequired),
        ('area_expertise', isRequired),
        ('module_name', isRequired),
        ('system_guidelines', isRequired),
        ('llm_model', isRequired),
        ('max_tokens', isRequired),
        ('documents_path', isRequired),
    ]

    data = get_data_from_request(request, fields)

    try:
        max_tokens = int(data.get("max_tokens")) 
    except:
        return jsonify({'error': 'Max tokens needs to be a number.'}), 400

    try: 
        id, chatbot_instance = ChatbotController.create_chatbot_instance(
            name = data.get("name"), 
            area_expertise = data.get("area_expertise"), 
            module_name = data.get("module_name"), 
            llm_model = data.get("llm_model"), 
            temperature = LLM_TEMPERATURE, 
            max_tokens = max_tokens, 
            documents_path = data.get("documents_path")
        )
        available_chatbots[id] = chatbot_instance

    except Exception as e:
        return jsonify({'error': "Something went wrong:" + str(e)}), 500

    return jsonify({'is_created': True, 'chatbot_id': id}), 201

@app.route('/api/chatbot/delete', methods=['POST', 'DELETE'])
def delete_chatbot():
    isRequired = True

    fields = [
        ('chatbot_id', isRequired),
    ]

    data = get_data_from_request(request, fields)
    
    if data.get('chatbot_id') not in available_chatbots.keys():
        return jsonify({'error': 'Chatbot not found.'}), 404

    try: 
        ChatbotController.delete_chatbot_instance(data.get('chatbot_id'))
        available_chatbots.pop(data.get('chatbot_id'), None)

    except Exception as e:
        return jsonify({'error': "Something went wrong:" + str(e)}), 500

    return jsonify({'is_deleted': True}, 200)

@app.route('/api/chatbot/update-settings', methods=['POST'])
def update_chatbot_settings():
    isRequired = True

    fields = [
        ('chatbot_id', isRequired),
        ('name', isRequired),
        ('area_expertise', isRequired),
        ('module_name', isRequired),
        ('system_guidelines', isRequired),
        ('llm_model', isRequired),
        ('max_tokens', isRequired),
    ]

    data = get_data_from_request(request, fields)
    
    if data.get('chatbot_id') not in available_chatbots.keys():
        return jsonify({'error': 'Chatbot not found.'}), 404

    try:
        chatbot_instance = ChatbotController.update_chatbot_instance_settings(
            id=data.get('chatbot_id'),
            name=data.get('name'),
            area_expertise=data.get('area_expertise'),
            module_name=data.get('module_name'),
            llm_model=data.get('llm_model'),
            system_guidelines=data.get('system_guidelines'),
            max_tokens=data.get('max_tokens')
        )
        available_chatbots[data.get('chatbot_id')] = chatbot_instance

    except Exception as e:
        return jsonify({'error': "Something went wrong:" + str(e)}), 500

    return jsonify({'is_updated': True}, 200)

@app.route('/api/chatbot/update-memory', methods=['POST'])
def update_chatbot_memory():
    isRequired = True

    fields = [
        ('chatbot_id', isRequired),
        ('deleted_documents', not isRequired),
        ('added_documents', not isRequired),
    ]

    data = get_data_from_request(request, fields)

    if data.get("chatbot_id") not in available_chatbots.keys():
        return jsonify({'error': 'Chatbot not found.'}), 404
    
    try:
        documents_not_updated = ChatbotController.update_chatbot_instance_memory(
            instance = available_chatbots[data.get("chatbot_id")],
            deleted_files = data.get('deleted_documents', []),
            added_files = data.get('added_documents', []),
        )
        
        if len(documents_not_updated) > 0:
            return jsonify({'error': f'Some documents could not be updated. The following documents were not updated: {", ".join(documents_not_updated)}'}), 400
    except Exception as e:
        return jsonify({'error': "Something went wrong:" + str(e)}), 500
    
    return jsonify({'is_updated': True}, 200)

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

def get_data_from_request(request, fields):
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict() or request.args.to_dict()

    required_fields = [
        (str(field), data.get(str(field))) for field, isRequired in fields if isRequired
    ]
    missing = [field for field, value in required_fields if not value]
    if missing:
        return jsonify({'error': f"Please provide all mandatory parameters. Missing parameters: {', '.join(missing)}"}), 400

    return data

if __name__ == "__main__":
    app.run()