from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

from api.settings import CHATBOT_TOOL_SYSTEM_PROMPT
from chatbot.tools import Tools

class RagGenerator:
    def __init__(self, model, temperature, num_predict):
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
        )
        self.tools = Tools.import_tools()

    def invoke(self, prompt):
        response = self.llm.invoke(prompt)
        return response

    def stream(self, prompt):
        full_response = ""
        async def stream_generator():
            nonlocal full_response
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            tool_output = self.check_tool_call(full_response)
            full_response += tool_output
            yield tool_output
        return stream_generator()

    def check_tool_call(self, llm_response):
        """Check the LLM response for potential tool calls and handle them appropriately."""
        system_prompt = CHATBOT_TOOL_SYSTEM_PROMPT.format(tools=self.render_tools_text())
        tool_name = self.generator.invoke(
            [HumanMessage(content=f"LLM OUTPUT:\n{llm_response}\n\nInstructions:\n{CHATBOT_TOOL_SYSTEM_PROMPT}")]
        ).content

        if tool_name and tool_name != "None" and tool_name in self.tools:
            return self.tools[tool_name].invoke()

    
    def render_tools_text(self):
        return "\n".join([f"- Tool name: {tool.name} | Tool description: {tool.description}" for tool in self.tools])