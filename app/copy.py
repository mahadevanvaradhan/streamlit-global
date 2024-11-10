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
import tiktoken
import numpy as np
from streamlit_authenticator import Authenticate

import yaml
from yaml.loader import SafeLoader
with open('./.config/config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)


# try:
#     authenticator.login()
# except Exception as e:
#     st.error(e)

# Attempt to log in
login_result = authenticator.login('sidebar', 'main')

# Check if login_result is a tuple before unpacking
if login_result:
    name, authentication_status, username = login_result
else:
    name, authentication_status, username = None, None, None

# If login is successful
if authentication_status:
    authenticator.logout('Logout', 'main')
    st.write(f'Welcome *{name}*')
    # Display the main content of the page
    st.title('Main Content Here')
    # Add the rest of your main page content below
    
# If login is unsuccessful
elif authentication_status == False:
    st.error('Username/password is incorrect')
    
# If login credentials have not been entered
elif authentication_status is None:
    # st.warning('Please enter your username and password')
    pass

# Assuming st.session_state["authentication_status"] is set elsewhere in your app:
if "authentication_status" in st.session_state:
    if st.session_state["authentication_status"]:
        # If logout is initiated
        if authenticator.logout('Logout', 'main'):
            st.write('You have been logged out.')
            st.session_state["authentication_status"] = None  # Reset the login state
        else:
            # If the user is still logged in, display the main content
            st.write(f'Welcome *{st.session_state["name"]}*')
            st.title('Main Content Here')
            # Add the rest of your main page content below
    
    elif st.session_state["authentication_status"] == False:
        st.error('Username/password is incorrect')
    else:  # This covers the case where authentication_status is None
        # st.warning('Please enter your username and password')
        pass
else:
    # Handle case where authentication_status is not in session_state
    st.warning('Please log in to continue.')

class DocumentChunker:
    def __init__(self, chunk_size: int = 4000):
        self.chunk_size = chunk_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoder

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count."""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split into sentences (roughly)
        sentences = text.replace('\n', ' ').split('.')
        
        for sentence in sentences:
            sentence = sentence.strip() + '.'
            sentence_length = self.count_tokens(sentence)
            
            if sentence_length > self.chunk_size:
                # If single sentence is too long, split by words
                words = sentence.split()
                current_word_chunk = []
                current_word_length = 0
                
                for word in words:
                    word_length = self.count_tokens(word + ' ')
                    if current_word_length + word_length > self.chunk_size:
                        chunks.append(' '.join(current_word_chunk))
                        current_word_chunk = [word]
                        current_word_length = word_length
                    else:
                        current_word_chunk.append(word)
                        current_word_length += word_length
                
                if current_word_chunk:
                    chunks.append(' '.join(current_word_chunk))
            
            elif current_length + sentence_length > self.chunk_size:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

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
        self.chunker = DocumentChunker()
        
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

    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        try:
            # Check if text needs chunking
            if self.chunker.count_tokens(text) > 8000:  # Safe limit for embedding
                chunks = self.chunker.chunk_text(text)
                all_embeddings = []
                
                # Show progress bar for embedding generation
                progress_bar = st.progress(0)
                for i, chunk in enumerate(chunks):
                    response = self.openai_client.embeddings.create(
                        input=chunk,
                        model=model
                    )
                    all_embeddings.append(response.data[0].embedding)
                    # Update progress
                    progress_bar.progress((i + 1) / len(chunks))
                
                # Average all embeddings
                combined_embedding = np.mean(all_embeddings, axis=0)
                progress_bar.empty()
                return combined_embedding.tolist()
            else:
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
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None

def display_conversation():
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def main():
    # st.set_page_config(page_title="Quickstep Document Processor", layout="wide")
    st.title("Quickstep Document Processor")

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
            st.session_state.embeddings = None
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
                with st.spinner("Generating embeddings... This may take a while for large documents."):
                    embedding = llm_handler.get_embedding(content, selected_embedding_model)
                    if embedding:
                        st.session_state.embeddings = embedding
                        st.success(f"Embedding generated! (Length: {len(embedding)})")
                        
                        # Display first few dimensions of the embedding
                        with st.expander("View Embedding Preview"):
                            st.write(f"First 10 dimensions: {embedding[:10]}")

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