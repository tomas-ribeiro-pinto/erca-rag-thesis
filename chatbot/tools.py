class Tools:
    @staticmethod
    def create_email_button(email_address, subject, body):
        """When the user needs or may need to contact the module convenor as per your conversation, create a draft email with the given address, subject, and body."""
        return  f"<a class=\"btn btn-primary\" href=\"mailto:{email_address}?subject={subject}&body={body}\">Send Email</a>"
    
    @staticmethod
    def import_tools():
        return {
            "create_email_button": Tool(
                name="create_email_button",
                description="Creates an email button",
                invoke_function=Tools.create_email_button
            )
        }
    
class Tool:
    def __init__(self, name, description, invoke_function):
        self.name = name
        self.description = description
        self.invoke_function = invoke_function

    def invoke(self, *args, **kwargs):
        return self.invoke_function(*args, **kwargs)