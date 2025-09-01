import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deepeval.models.base_model import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
from components.chat_open_router import ChatOpenRouter

from api.controllers.chatbot_controller import ChatbotController
from api.models.chatbot import Chatbot

class EvaluationChatbot(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        use_ollama=False,
        keep_memory=False,
        use_rag=False
    ):
        chatbot_instance = dict(ChatbotController.get_chatbot_instance_by_id(1))
        chatbot_instance['llm_model'] = model
        chatbot_instance['use_ollama'] = use_ollama
        self.model_name = model
        self.use_rag = use_rag
        self.keep_memory = keep_memory
        
        if use_rag:
            self.model = Chatbot(chatbot_instance, keep_memory=keep_memory)
        else:
            # Use the appropriate LLM based on use_ollama setting
            if use_ollama:
                self.model = self.llm = ChatOllama(
                    model=self.model_name,
                    temperature=chatbot_instance['temperature'],
                    num_predict=chatbot_instance['max_tokens'],
                )
            else:
                self.model = self.llm = ChatOpenRouter(
                    model_name=self.model_name,
                    temperature=chatbot_instance['temperature'],
                    max_tokens=chatbot_instance['max_tokens'],
                )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        if self.use_rag and not self.keep_memory:
            result, context = chat_model.invoke(prompt)
            return result, context
        else:
            result = chat_model.invoke(prompt).content
        return str(result)

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name