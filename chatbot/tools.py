from langchain_core.messages import HumanMessage

class Tools:
    @staticmethod
    def create_email_button(llm, last_message):
        system_prompt = ("According to the last output message create a draft email subject, body and output a button in this format:\n" \
        "I have prepared you a email draft you can use by clicking in the button below:\n"
        "<a class=\"btn btn-primary\" href=\"mailto:{email_address}?subject={subject}&body={body}\">Send Email</a>")
        output = llm.invoke(
            [HumanMessage(content=f"LLM OUTPUT:\n{last_message}\n\nInstructions:\n{system_prompt}")]
        ).content
        return output
    
    @staticmethod
    def import_tools():
        return [
            Tool(
                name="create_email_button",
                description="Creates a draft email button with subject and body based on the last message content. " \
                    "If the user wants or might benefit of contacting the module convenor or any other person, create an email button.",
                invoke_function=Tools.create_email_button
            )
        ]
    
class Tool:
    def __init__(self, name, description, invoke_function):
        self.name = name
        self.description = description
        self.invoke_function = invoke_function

    def invoke(self, llm, last_message):
        return self.invoke_function(llm, last_message)