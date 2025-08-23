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

            tool_output = self.check_tool_call(f"PROMPT: {prompt} | YOUR RESPONSE:{full_response}")
            if tool_output:  # Only add and yield if there's actual tool output
                full_response += "\n" + tool_output
                yield tool_output
        return stream_generator()

    def check_tool_call(self, llm_response):
        """Check the LLM response for potential tool calls and handle them appropriately."""
        system_prompt = CHATBOT_TOOL_SYSTEM_PROMPT.format(tools=self.render_tools_text(), last_message=llm_response)
        tool_number = self.llm.invoke(
            [HumanMessage(content=system_prompt)]
        ).content

        print(f"Tool number identified: {tool_number}")

        if tool_number and tool_number != "None" and 0 < int(tool_number) <= len(self.tools):
            return self.tools[int(tool_number) - 1].invoke(self.llm, llm_response)

        return ""

    
    def render_tools_text(self):
        return "\n".join([f"- Tool number: {i+1} | Tool description: {tool_obj.description}" for i, tool_obj in enumerate(self.tools)])