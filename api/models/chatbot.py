"""
================================================================================
RAG Chatbot API for Education - Thesis Project
--------------------------------------------------------------------------------
Author: Tomás Pinto
Date: August 2025
Description:
    This file implements a chatbot class that interacts with various components
    such as the RAG generator, retriever, and the language model. It manages
    the workflow of processing user messages and generating responses.
    It uses the LangChain orchestration framework as a skeleton for building
    the chatbot's functionality.
================================================================================
"""

import asyncio
from api.controllers.user_controller import UserController
from api.settings import CHATBOT_SYSTEM_PROMPT, MAX_MESSAGES, CHATBOT_SUMMARY_SYSTEM_PROMPT
from chatbot.rag_generator import RagGenerator
from chatbot.rag_retriever import RagRetriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, RemoveMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph

import sqlite3
import os

class Chatbot:
    def __init__(self, instance, chatbot_api_db_path="./databases/chatbot_instances.db"):
        self.chatbot_id = instance["chatbot_id"]
        self.llm = ChatOllama(
            model = instance["llm_model"],
            temperature = instance["temperature"],
            num_predict = instance["num_predict"],
        )

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        vector_db_path = os.path.join(project_root, instance["vector_db_path"])
        self.retriever = RagRetriever(vector_db_path=vector_db_path)
        self.generator = RagGenerator(self.llm)
        self.user_history_db_path = os.path.join(project_root, chatbot_api_db_path)

        system_prompt = instance["system_prompt"] if "system_prompt" in instance else CHATBOT_SYSTEM_PROMPT
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("user", "{user_prompt}")]
        )
        
        # Initialize the workflow with a state graph (without compiling)
        self.workflow = StateGraph(state_schema=MessagesState)
        
        # Define the node and edge
        self.workflow.add_node("model", self.get_state)
        self.workflow.add_edge(START, "model")
        
        # Store user-specific compiled apps
        self.user_apps = {}

    def get_state(self, state: MessagesState):
        """Simple placeholder method that returns the state unchanged"""
        return state

    def get_user_app(self, user_id):
        """Get or create a user-specific compiled workflow"""
        if user_id not in self.user_apps:
            # Create a new memory checkpointer for this user
            memory = MemorySaver()
            self.user_apps[user_id] = self.workflow.compile(checkpointer=memory)
        return self.user_apps[user_id]

    def prepare_streaming_context(self, state: MessagesState):
        """
            Prepares the context and messages required for streaming responses.
            Retrieves relevant documents for the latest user message, manages conversation history,
            and summarises if the history exceeds the maximum allowed messages.
            Returns a dictionary containing:
                - messages_for_llm: List of messages to send to the language model.
                - state_updates: Messages to update the conversation state (if summarisation occurs).
                - user_prompt: The latest user message.
        """

        # Get the last user message
        user_message = state["messages"][-1].content

        # Retrieve relevant documents to user_message
        docs = self.retriever.invoke(user_message, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Filter messages in state to only contain conversation messages (no system messages)
        conversation_history = [
            msg for msg in state["messages"] 
            if isinstance(msg, (HumanMessage, AIMessage))
        ]

        # If conversation history is too long, summarize it
        if len(conversation_history) >= MAX_MESSAGES:
            # Clean history for summarization by removing metadata
            clean_history = [
                {"role": "user" if isinstance(msg, HumanMessage) else "assistant", 
                "content": msg.content}
                for msg in conversation_history
            ]
    
            # Generate summary 
            summary_message = self.llm.invoke(
                [HumanMessage(content=f"Conversation:\n{clean_history}\n\nInstructions:\n{CHATBOT_SUMMARY_SYSTEM_PROMPT}")]
            )

            # Try to safely create delete messages, but have a fallback approach
            delete_messages = []
            safe_delete = True
            
            for m in state["messages"]:
                if hasattr(m, 'id') and m.id is not None:
                    try:
                        delete_messages.append(RemoveMessage(id=m.id))
                    except Exception as e:
                        # If any message deletion fails, fall back to a simpler approach
                        print(f"Warning: Could not create RemoveMessage for message ID {getattr(m, 'id', 'None')}: {str(e)}")
                        continue
            
            # Re-add user message
            human_message = HumanMessage(content=user_message)

            # Format the prompt_template with retrieved context and user message
            formatted_messages = self.prompt_template.format_messages(context=context, user_prompt=user_message)
            messages_for_llm = [summary_message, human_message] + formatted_messages
            
            return {
                "messages_for_llm": messages_for_llm,
                "state_updates": [summary_message, human_message] + delete_messages,
                "user_prompt": user_message
            }
        else:
            # Format the prompt with context for the current conversation
            formatted_messages = self.prompt_template.format_messages(context=context, user_prompt=user_message)
            # Use only the system message from formatted_messages and keep conversation history
            system_message = formatted_messages[0]  # The system message with context
            messages_for_llm = [system_message] + state["messages"]
            
            return {
                "messages_for_llm": messages_for_llm,
                "state_updates": None,
                "user_prompt": user_message
            }

    
    async def get_stream_response(self, user_id, user_prompt, thread_id="default"):
        """Get response as a stream using the LangGraph workflow with call_model_streaming"""

        user = UserController.get_user_by_id(user_id)

        if user['username'] != "guest_user":
            UserController.add_user_history_entry(user_id, self.chatbot_id, str(user_prompt), "user")

        try:
            # Get user-specific app with dedicated memory for context
            app = self.get_user_app(user_id)
            config = {"configurable": {"thread_id": thread_id}}

            # Get current state to build context with new user message
            current_state = app.get_state(config)
            existing_messages = current_state.values.get("messages", []) if current_state.values else []
            
            # Add the new user message
            user_message = HumanMessage(content=user_prompt)
            all_messages = existing_messages + [user_message]
            state_with_user_message = {"messages": all_messages}

            # Prepare streaming context
            streaming_data = self.prepare_streaming_context(state_with_user_message)
            messages_for_llm = streaming_data["messages_for_llm"]
            state_updates = streaming_data["state_updates"]
            
            full_response = ""
            async def stream_generator():
                nonlocal full_response
                try:
                    # Use the LLM's streaming
                    for chunk in self.llm.stream(messages_for_llm):
                        if hasattr(chunk, 'content') and chunk.content:
                            full_response += chunk.content
                            yield chunk.content
                    
                    # Create the AI response message
                    response_message = AIMessage(content=full_response)

                    # Update state based on whether we had summarisation or not
                    if state_updates:
                        final_state = {"messages": state_updates + [response_message]}
                    else:
                        final_state = {"messages": all_messages + [response_message]}
                    
                    # Update the LangGraph state
                    app.update_state(config, final_state)
                    
                    # Save to user history
                    if user['username'] != "guest_user":
                        UserController.add_user_history_entry(user_id, self.chatbot_id, str(full_response), "assistant")
                        
                except Exception as streaming_error:
                    yield f"Error during streaming: {str(streaming_error)}"
                    
            return stream_generator()
            
        except Exception as setup_error:
            error = f"Error setting up streaming: {str(setup_error)}"
            async def error_generator():
                nonlocal error
                yield error
            return error_generator()