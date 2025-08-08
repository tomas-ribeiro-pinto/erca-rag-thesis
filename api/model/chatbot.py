import asyncio
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

        system_template = (
            "You are an expert assistant helping a university student with questions about image processing. "
            "Use the provided context from lecture notes, slides, and other materials to answer clearly and concisely. "
            "If you can't answer or it is out of context, say 'I don't know'."
            "Ensure your response is suitable for a student, informative, and does not exceed 4096 tokens."
            "Context: {context}"
        )
        self.prompt_template = ChatPromptTemplate.from_messages(
            [("system", system_template), ("user", "{user_prompt}")]
        )
        
        # Initialize the workflow with a state graph
        #TODO: Get state from db
        self.workflow = StateGraph(state_schema=MessagesState)
        
        # Define the node and edge
        self.workflow.add_node("model", self.call_model)
        self.workflow.add_edge(START, "model")
        
        # Add simple in-memory checkpointer
        memory = MemorySaver()
        self.app = self.workflow.compile(checkpointer=memory)

    def call_model(self, state: MessagesState):
        """Define the function that calls the model with RAG context"""
        # Get the last user message
        user_message = state["messages"][-1].content
        
        # Retrieve relevant documents
        docs = self.retriever.invoke(user_message, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create system prompt with context
        system_prompt = (
            "You are an expert assistant helping university students with questions about image processing. "
            "Use the provided context from lecture notes, slides, and other materials to answer questions clearly and concisely.\n\n"
            "Guidelines:\n"
            "1. Always be polite, patient, and supportive\n"
            "2. Break down complex concepts into understandable parts\n"
            "3. Provide examples when helpful\n"
            "4. If the question is unclear, ask for clarification\n"
            "5. When you can't answer based on the provided context:\n"
            "   - Clearly state you don't know\n"
            "   - Suggest specific lecturer(s) to contact for this topic\n"
            "   - Provide their email if available\n"
            "   - Recommend office hours if appropriate\n\n"
            "Remember to keep responses under 4096 tokens and suitable for university students."
            f"Current Context:\n{context}\n\n"
        )

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
            
            # Call the model with summary & response
            response = self.llm.invoke([SystemMessage(content=system_prompt), summary_message, human_message])
            message_updates = [summary_message, human_message, response] + delete_messages
        else:
            message_updates = self.llm.invoke([SystemMessage(content=system_prompt)] + state["messages"])

        return {"messages": message_updates}
    
    async def get_response(self, user_id, user_prompt, thread_id="default"):
        """Get response using the LangGraph workflow"""
        config = {"configurable": {"thread_id": thread_id}}
        
        # Create input state with the user message
        input_state = {"messages": [{"role": "user", "content": user_prompt}]}

        self.add_message_to_history(user_id, str(user_prompt), "user")

        # Invoke the workflow
        result = await asyncio.to_thread(self.app.invoke, input_state, config)
        
        output = result["messages"][-1].content

        self.add_message_to_history(user_id, str(output), "assistant")

        # Return the last message content
        return output
    
    def add_message_to_history(self, user_id, content, role):
        con = sqlite3.connect(self.user_history_db_path)
        cur = con.cursor()
        cur.execute('''
            INSERT INTO user_history (user_id, chatbot_id, content, role)
            VALUES (?, ?, ?, ?)
        ''', (user_id, self.chatbot_id, content, role))
        con.commit()
        con.close()