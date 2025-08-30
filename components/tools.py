from typing import List
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
import os

@tool
def output_email_button(email_address: str = "", subject: str = "", body: str = "", button_placeholder: str = "Send email", button_pre_text: str = "\n*You can use the following button to send an email:*\n\n") -> AIMessage:
    """Creates a draft email button with subject and body. ONLY use this tool when the user asks to contact someone or may benefit from contacting someone, needs help with coursework submission, has questions about grades, or needs to report technical issues. Do not use for general module questions that can be answered from the course materials."""

    output = f"\n\n{button_pre_text} <a class=\"email-btn\" href=\"mailto:{email_address}?subject={subject}&body={body}\">{button_placeholder}</a>"
    
    return AIMessage(
        content = output
    )

@tool
def output_context_reference(cited_sources: List[str] = None) -> AIMessage:
    """Use this tool to reference specific course materials, lectures, or lab sessions from retrieved context when answering academic questions. Use when the answer directly relates to specific course content that should be cited, include the name of the file material."""

    if not cited_sources:
        return AIMessage(content="")
    
    output = "\n\n**Cited Materials:**\n\n"
    
    for source in cited_sources:
        # Extract filename from path and create a display name
        filename = os.path.basename(source)
        display_name = filename.replace('.pdf', '').replace('_', ' ').replace('-', ' ')
        
        # Get base URL from environment variable or use default
        base_url = os.getenv('BASE_URL', '')
        document_url = f"{base_url}/documents/{filename}"

        output += f'<a href="{document_url}" target="_blank">{display_name}</a>\n\n'

    return AIMessage(
        content = output
    )