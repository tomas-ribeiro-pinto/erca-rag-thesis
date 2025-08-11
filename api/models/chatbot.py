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
from api.settings import CHATBOT_SYSTEM_TEMPLATE
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

        # Use a separate database file for the API to avoid conflicts
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        api_db_path = os.path.join(project_root, instance["vector_db_path"])
        self.retriever = RagRetriever(db_path=api_db_path)
        self.generator = RagGenerator(self.llm)
        self.user_history_db_path = os.path.join(project_root, chatbot_api_db_path)

        system_template = CHATBOT_SYSTEM_TEMPLATE
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{user_prompt}")]
        )
        
        # Initialize the workflow with a state graph (without compiling)
        self.workflow = StateGraph(state_schema=MessagesState)
        
        # Define the node and edge
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")
        
        # Store user-specific compiled apps
        self.user_apps = {}

    def get_user_app(self, user_id):
        """Get or create a user-specific compiled workflow"""
        if user_id not in self.user_apps:
            # Create a new memory checkpointer for this user
            memory = MemorySaver()
            self.user_apps[user_id] = self.workflow.compile(checkpointer=memory)
        return self.user_apps[user_id]

    def call_model(self, state: MessagesState):
        """Define the function that calls the model with RAG context"""
        # Get the last user message
        user_message = state["messages"][-1].content
        
        # Retrieve relevant documents
        docs = self.retriever.invoke(user_message, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Filter to only conversation messages (no system messages)
        conversation_history = [
            msg for msg in state["messages"] 
            if isinstance(msg, (HumanMessage, AIMessage))
        ]

        # If conversation history is too long, summarize it
        if len(conversation_history) >= 8:
            # Prepare clean history for summarization (remove metadata)
            clean_history = [
                {"role": "user" if isinstance(msg, HumanMessage) else "assistant", 
                "content": msg.content}
                for msg in conversation_history
            ]
        
            # Generate summary
            summary_prompt = (
                "Create a precise technical summary of this image processing conversation. "
                "Include: key questions, solutions, and decisions. "
                "Omit greetings and social remarks. But do include any user preferences or important details such as name."
                "Maximum 150 words. Third-person perspective."
            )
        
            summary_message = self.llm.invoke(
                [HumanMessage(content=f"Conversation:\n{clean_history}\n\nInstructions:\n{summary_prompt}")]
            )

            # Delete messages that we no longer want to show up
            delete_messages = [RemoveMessage(id=m.id) for m in state["messages"]]
            # Re-add user message
            human_message = HumanMessage(content=user_message)
            
            # Format the prompt with context and invoke the model
            formatted_messages = self.prompt_template.format_messages(context=context, user_prompt=user_message)
            response = self.llm.invoke([summary_message, human_message] + formatted_messages)
            message_updates = [summary_message, human_message, response] + delete_messages
        else:
            # Format the prompt with context for the current conversation
            formatted_messages = self.prompt_template.format_messages(context=context, user_prompt=user_message)
            # Use only the system message from formatted_messages, keep conversation history
            system_message = formatted_messages[0]  # The system message with context
            response = self.llm.invoke([system_message] + state["messages"])
            message_updates = response

        return {"messages": message_updates}
    
    async def get_response(self, user_id, user_prompt, thread_id="default"):
        """Get response using the LangGraph workflow with user-specific memory"""
        # Get user-specific app with dedicated memory
        app = self.get_user_app(user_id)
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create input state with the user message
        input_state = {"messages": [{"role": "user", "content": user_prompt}]}

        user = UserController.get_user_by_id(user_id)

        if user['username'] != "guest_user":
            UserController.add_user_history_entry(user_id, self.chatbot_id, str(user_prompt), "user")

        # Invoke the user-specific workflow
        result = await asyncio.to_thread(app.invoke, input_state, config)
        
        output = result["messages"][-1].content

        if user['username'] != "guest_user":
            UserController.add_user_history_entry(user_id, self.chatbot_id, str(output), "assistant")

        # Return the last message content
        return output