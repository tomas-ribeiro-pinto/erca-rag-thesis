from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

import threading
import queue

@tool
def output_email_button(email_address="", subject="", body="", button_placeholder="Send email", button_pre_text="\n*You can use the following button to send an email:*\n\n"):
    """Creates a draft email button with subject and body. ONLY use this tool when the user asks to contact someone or may benefit from contacting someone, needs help with coursework submission, has questions about grades, or needs to report technical issues. Do not use for general module questions that can be answered from the course materials."""

    output = f"\n\n{button_pre_text} <a class=\"send-btn\" href=\"mailto:{email_address}?subject={subject}&body={body}\">{button_placeholder}</a>"
    
    return AIMessage(
        content = output
    )

@tool
def output_context_reference(materials_discussed=""):
    """Use this tool to reference specific course materials, lectures, or lab sessions from retrieved context when answering academic questions. Use when the answer directly relates to specific course content that should be cited, include the name of the material and the relevant details such as the number of the slide and lecture."""

    output = f"\n\n*Materials discussed:*\n\n{materials_discussed}"

    return AIMessage(
        content = output
    )

class RagGenerator:
    def __init__(self, model, temperature, num_predict):
        self.tools = [output_email_button, output_context_reference]
        self.llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
        )
        # Tool LLM with tools for tool calling
        self.tool_llm = ChatOllama(
            model=model,
            temperature=temperature,
            num_predict=num_predict,
        ).bind_tools(self.tools)

    def invoke(self, prompt):
        response = self.llm.invoke(prompt)
        return response
    
    def stream(self, prompt):
        # Create a queue to communicate between threads
        tool_result_queue = queue.Queue()
        
        t1 = threading.Thread(target=self.check_tool_calling, args=(prompt, tool_result_queue))
        t1.start()

        full_response = ""
        async def stream_generator():
            nonlocal full_response
            # First, stream the conversational response
            for chunk in self.llm.stream(prompt):
                if hasattr(chunk, 'content') and chunk.content:
                    full_response += chunk.content
                    yield chunk.content

            # Wait for the tool checking thread to complete
            t1.join()
            
            # Check if there's a tool result to append
            try:
                tool_result = tool_result_queue.get_nowait()
                if tool_result and hasattr(tool_result, 'content'):
                    yield tool_result.content
            except queue.Empty:
                # No tool result available
                pass

        return stream_generator()
    
    def check_tool_calling(self, prompt, result_queue):
        # After the main response, check if tools should be called
        try:
            tool_response = self.tool_llm.invoke(prompt)
            if hasattr(tool_response, 'tool_calls') and tool_response.tool_calls:
                # Execute the tool calls
                for tool_call in tool_response.tool_calls:
                    if tool_call['name'] == 'output_email_button':
                        args = tool_call['args']
                        tool_result = output_email_button.invoke(args)
                        result_queue.put(tool_result)
                        return
                    elif tool_call['name'] == 'output_context_reference':
                        args = tool_call['args']
                        tool_result = output_context_reference.invoke(args)
                        result_queue.put(tool_result)
                        return
            # If no tools were called, put None in the queue
            result_queue.put(None)
        except Exception as e:
            print(f"Tool execution error: {e}")
            result_queue.put(None)