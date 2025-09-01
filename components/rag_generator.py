import asyncio
import threading
import queue

from langchain_ollama import ChatOllama

from components.chat_open_router import ChatOpenRouter
from components.tools import output_email_button, output_context_reference

class RagGenerator:
    def __init__(self, model, temperature, num_predict, use_ollama=False):
        self.tools = [output_email_button, output_context_reference]
        if use_ollama:
            self.llm = ChatOllama(
                model=model,
                temperature=temperature,
            num_predict=num_predict,
        )
        else:
            self.llm = ChatOpenRouter(
                model_name=model,
                temperature=temperature,
                max_tokens=num_predict,
            )

        # Tool LLM with tools for tool calling
        self.tool_llm = self.llm.bind_tools(self.tools)

    def invoke(self, prompt, async_mode=False):
        # Create a queue to communicate between threads
        tool_result_queue = queue.Queue()
        
        t1 = threading.Thread(target=self.check_tool_calling, args=(prompt, tool_result_queue))
        t1.start()

        response = self.llm.invoke(prompt).content
        
        # Wait for the tool checking thread to complete
        t1.join()
        
        # Check if there's a tool result to append
        try:
            tool_result = tool_result_queue.get_nowait()
            if tool_result and hasattr(tool_result, 'content'):
                response += tool_result.content
        except queue.Empty:
            # No tool result available
            pass

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
                        # Debug print to see the actual structure
                        print(f"Tool call args: {args}")
                        
                        # Handle case where cited_sources might be in schema format
                        if 'cited_sources' in args and isinstance(args['cited_sources'], dict):
                            if 'items' in args['cited_sources']:
                                args['cited_sources'] = args['cited_sources']['items']
                            elif 'value' in args['cited_sources']:
                                args['cited_sources'] = args['cited_sources']['value']
                            else:
                                # If it's just a type definition, skip this tool call
                                print(f"Skipping tool call with schema-only args: {args}")
                                result_queue.put(None)
                                return
                        tool_result = output_context_reference.invoke(args)
                        result_queue.put(tool_result)
                        return
            # If no tools were called, put None in the queue
            result_queue.put(None)
        except Exception as e:
            print(f"Tool execution error: {e}")
            result_queue.put(None)