import streamlit as st
import anthropic
import openai
from typing import Optional, List, Dict
import docx
import json
import PyPDF2
import io
import os
from datetime import datetime

class DocumentProcessor:
    @staticmethod
    def read_file_content(file) -> Optional[str]:
        if file is None:
            return None
        
        file_extension = file.name.split('.')[-1].lower()
        content = None
        
        try:
            file_bytes = io.BytesIO(file.read())
            file_bytes.seek(0)
            
            if file_extension in ['txt', 'md']:
                content = file_bytes.read().decode('utf-8')
            
            elif file_extension == 'pdf':
                pdf_reader = PyPDF2.PdfReader(file_bytes)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text()
            
            elif file_extension in ['doc', 'docx']:
                try:
                    doc = docx.Document(file_bytes)
                    paragraphs = []
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            paragraphs.append(paragraph.text.strip())
                    content = "\n".join(paragraphs)
                except Exception as docx_error:
                    st.error(f"Error processing DOCX file: {str(docx_error)}")
                    return None
            
            elif file_extension == 'json':
                content = json.loads(file_bytes.read().decode('utf-8'))
                content = json.dumps(content, indent=2)
            
            file.seek(0)
            return content
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None

class LLMHandler:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
        self.openai_client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        self.claude_models = {
            "claude-3-opus-20240229": "Claude 3 Opus",
            "claude-3-sonnet-20240229": "Claude 3 Sonnet",
            "claude-3-haiku-20240307": "Claude 3 Haiku",
            "claude-2.1": "Claude 2.1"
        }
        
        self.embedding_models = {
            "text-embedding-3-small": "OpenAI Ada 3 Small",
            "text-embedding-3-large": "OpenAI Ada 3 Large",
            "text-embedding-ada-002": "OpenAI Ada 2"
        }

    def get_llm_response(self, messages: List[Dict], system_prompt: str, model: str) -> str:
        try:
            if model in ["claude-2.1"]:
                response = self.anthropic_client.messages.create(
                    max_tokens=1000,
                    model=model,
                    messages=messages,
                    system=system_prompt
                )
            else:
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=messages,
                    system=system_prompt
                )
            return response.content[0].text
        except Exception as e:
            st.error(f"Error getting LLM response: {str(e)}")
            return None

    def get_embedding(self, text: str, model: str) -> list:
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model=model
            )
            return response.data[0].embedding
        except Exception as e:
            st.error(f"Error getting embedding: {str(e)}")
            return None

def initialize_session_state():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'document_context' not in st.session_state:
        st.session_state.document_context = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []

def display_conversation():
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def main():
    st.set_page_config(page_title="Interactive LLM Document Processor", layout="wide")
    st.title("Interactive LLM Document Processor")

    # Initialize session state
    initialize_session_state()

    # Initialize handlers
    doc_processor = DocumentProcessor()
    llm_handler = LLMHandler()

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # System prompt input
        system_prompt = st.text_area(
            "System Prompt",
            value="""You are a helpful AI assistant. Answer questions based on the provided document.
            When the document content is not relevant to the question, let the user know.
            Keep your responses concise and relevant.""",
            height=100
        )
        
        # Model selection
        selected_claude_model = st.selectbox(
            "Select Claude Model",
            options=list(llm_handler.claude_models.keys()),
            format_func=lambda x: llm_handler.claude_models[x]
        )
        
        selected_embedding_model = st.selectbox(
            "Select Embedding Model",
            options=list(llm_handler.embedding_models.keys()),
            format_func=lambda x: llm_handler.embedding_models[x]
        )

        # Clear conversation button
        if st.button("Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.messages = []
            st.rerun()

    # Document upload section
    st.header("Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document (PDF, DOCX, TXT, MD, JSON)",
        type=['pdf', 'docx', 'txt', 'md', 'json']
    )

    if uploaded_file:
        content = doc_processor.read_file_content(uploaded_file)
        if content:
            st.session_state.document_context = content
            st.success("File processed successfully!")
            
            with st.expander("View Document Content"):
                st.text_area("Content", value=content, height=200, disabled=True)

            # Get embedding if requested
            if st.button("Generate Embedding"):
                embedding = llm_handler.get_embedding(content, selected_embedding_model)
                if embedding:
                    st.success(f"Embedding generated! (Length: {len(embedding)})")

    # Chat interface
    st.header("Chat Interface")
    display_conversation()

    # User input
    user_input = st.chat_input("Ask a question about the document...")
    
    if user_input:
        # Add user message to conversation
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
        
        # Prepare messages for API
        messages = []
        if st.session_state.document_context:
            messages.append({
                "role": "user",
                "content": f"Here is the document content:\n{st.session_state.document_context}"
            })
        
        # Add conversation history
        for msg in st.session_state.conversation_history:
            messages.append(msg)

        # Get AI response
        ai_response = llm_handler.get_llm_response(
            messages=messages,
            system_prompt=system_prompt,
            model=selected_claude_model
        )

        if ai_response:
            # Add AI response to conversation
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": ai_response
            })
            
            # Rerun to update the display
            st.rerun()

if __name__ == "__main__":
    main()