import asyncio
from chatbot.rag_generator import RagGenerator
from chatbot.rag_retriever import RagRetriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
import os

class Chatbot:
    def __init__(self):
        llm = ChatOllama(
            model = "llama3.1",
            temperature = 0.8,
            num_predict = 4096,
        )
        # Use a separate database file for the API to avoid conflicts
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        api_db_path = os.path.join(project_root, "api_rag_milvus.db")
        self.retriever = RagRetriever(db_path=api_db_path)
        self.generator = RagGenerator(llm)

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
    
    async def get_response(self, user_prompt):
        docs = self.retriever.invoke(user_prompt, k=5)
        prompt = self.prompt_template.invoke({"user_prompt": user_prompt, "context": "\n\n".join([doc.page_content for doc in docs])})
        response = await asyncio.to_thread(self.generator.invoke, prompt)
        return response.content
