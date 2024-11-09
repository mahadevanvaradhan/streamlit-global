import streamlit as st
import anthropic
import openai
from typing import Optional
import docx
import json
import PyPDF2
import io
import os


OPEN_API_KEY = os.getenv("OPEN_API_KEY")
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

class DocumentProcessor:
    @staticmethod
    def read_file_content(file) -> Optional[str]:
        if file is None:
            return None
        
        file_extension = file.name.split('.')[-1].lower()
        content = None
        
        try:
            # Store the file content in memory
            file_bytes = io.BytesIO(file.read())
            # Reset the file pointer for subsequent reads
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
                        if paragraph.text.strip():  # Only add non-empty paragraphs
                            paragraphs.append(paragraph.text.strip())
                    content = "\n".join(paragraphs)
                except Exception as docx_error:
                    st.error(f"Error processing DOCX file: {str(docx_error)}")
                    return None
            
            elif file_extension == 'json':
                content = json.loads(file_bytes.read().decode('utf-8'))
                content = json.dumps(content, indent=2)
            
            # Reset file pointer for future reads
            file.seek(0)
            return content
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None

class LLMHandler:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.openai_client = openai.OpenAI(api_key=OPEN_API_KEY)
        
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

    def get_llm_response(self, prompt: str, system_prompt: str, model: str) -> str:
        try:
            if model in ["claude-2.1"]:
                # Old format for Claude 2.1
                response = self.anthropic_client.messages.create(
                    max_tokens=1000,
                    model=model,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    system=system_prompt
                )
            else:
                # New format for Claude 3 models
                response = self.anthropic_client.messages.create(
                    model=model,
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
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

def main():
    st.set_page_config(page_title="LLM Document Processor", layout="wide")
    st.title("LLM Document Processor")

    # Initialize handlers
    doc_processor = DocumentProcessor()
    llm_handler = LLMHandler()

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        
        # System prompt input
        system_prompt = st.text_area(
            "System Prompt",
            value="You are a helpful AI assistant. Answer questions based on the provided document.",
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

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a document (PDF, DOCX, TXT, MD, JSON)",
            type=['pdf', 'docx', 'txt', 'md', 'json']
        )

        if uploaded_file:
            # Store the file content in session state to avoid rereading
            if 'file_content' not in st.session_state:
                content = doc_processor.read_file_content(uploaded_file)
                if content:
                    st.session_state['file_content'] = content
            
            if 'file_content' in st.session_state:
                st.success("File processed successfully!")
                st.text_area("Document Content", value=st.session_state['file_content'], height=300, disabled=True)

                # Get embedding if requested
                if st.button("Generate Embedding"):
                    embedding = llm_handler.get_embedding(st.session_state['file_content'], selected_embedding_model)
                    if embedding:
                        st.success(f"Embedding generated! (Length: {len(embedding)})")

    with col2:
        st.header("Ask Questions")
        user_question = st.text_area("Enter your question about the document", height=100)

        if st.button("Get Answer"):
            if not uploaded_file:
                st.warning("Please upload a document first!")
            elif not user_question:
                st.warning("Please enter a question!")
            else:
                if 'file_content' in st.session_state:
                    prompt = f"""Here is the document content:
                    {st.session_state['file_content']}
                    
                    Question: {user_question}
                    
                    Please provide an answer based on the document content."""

                    response = llm_handler.get_llm_response(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=selected_claude_model
                    )

                    if response:
                        st.markdown("### Answer")
                        st.write(response)

if __name__ == "__main__":
    main()