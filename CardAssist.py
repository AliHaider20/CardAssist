import asyncio
import os
import streamlit as st
from dotenv import load_dotenv
from tqdm import tqdm
import faiss
import numpy as np
import tempfile

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions import kernel_function

from typing import Annotated, List

# Load environment variables
load_dotenv()

search_api = os.getenv("AI_SEARCH_API")
search_endpoint = "https://cardassist.search.windows.net"
ai_foundry_api = os.getenv("AI_FOUNDRY_MODEL_API")
llm_endpoint = os.getenv("LLM_ENDPOINT")

# Initialize embedding model globally
embedding_service = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Session state initialization
if "kernel" not in st.session_state:
    st.session_state.kernel = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "faiss_index" not in st.session_state:
    st.session_state.faiss_index = None
if "docs" not in st.session_state:
    st.session_state.docs = []
if "loading" not in st.session_state:
    st.session_state.loading = False
if "error" not in st.session_state:
    st.session_state.error = ""

# PDF processing functions
async def load_and_process_pdf_async(pdf_path: str) -> List[str]:
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=30)
        md_docs = text_splitter.split_documents(documents)
        md_docs = [doc.page_content for doc in md_docs]
        return md_docs
    except Exception as e:
        raise RuntimeError(f"Failed to process PDF: {e}")

def generate_embeddings(docs: List[str]) -> np.ndarray:
    embeddings = []
    for doc in tqdm(docs, desc="Generating embeddings"):
        emb = embedding_service.encode([doc])
        embeddings.append(emb[0])
    return np.array(embeddings).astype("float32")

def create_faiss_index(embeddings: np.ndarray):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# Credit Card Plugin for Semantic Kernel
class CreditCardPlugin:
    def __init__(self, faiss_index, docs):
        self.index = faiss_index
        self.docs = docs

    @kernel_function(description="Deactivate a credit card; returns a confirmation message")
    async def deactivate_card(self, card_number: Annotated[str, "The credit card number to deactivate"]) -> str:
        print("Function called to deactivate card")
        return f"Credit card {card_number} has been deactivated."

    @kernel_function(description="Activate a credit card; returns a confirmation message")
    async def activate_card(self, card_number: Annotated[str, "The credit card number to activate"]) -> str:
        print("Function called to activate card")
        return f"Credit card {card_number} has been activated."

    @kernel_function(description="Get card information and other general information about the card and account management")
    async def rag_query(self, query: Annotated[str, "The user query for RAG (Retrieval-Augmented Generation)"]) -> str:
        print("Function called for RAG query")
        query_embedding = embedding_service.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k=5)
        relevant_chunks = [self.docs[i] for i in indices[0]]
        context = "\n".join(relevant_chunks)
        return f"{context}\n\nUser Query: {query}"

# Setup Semantic Kernel
def setup_kernel(faiss_index, docs):
    model_id = "gpt-4o-mini"
    kernel = Kernel()
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=model_id,
            endpoint=llm_endpoint,
            api_key=ai_foundry_api
        )
    )
    kernel.add_plugin(CreditCardPlugin(faiss_index, docs), plugin_name="CreditCard")
    return kernel

# Chat handling
async def process_message(message: str) -> str:
    kernel: Kernel = st.session_state.kernel
    if kernel is None:
        return "System is not ready. Please upload a PDF first."

    arguments = KernelArguments(
        settings=AzureChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            top_p=0.9,
            temperature=0,
        )
    )

    try:
        response = await kernel.invoke_prompt(message, arguments=arguments)
        return response.value[0].content
    except Exception as e:
        return f"Error: {str(e)}"

# Handle PDF upload
async def handle_pdf_upload(file):
    st.session_state.loading = True
    st.session_state.error = ""
    try:
        if file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            temp_file.write(file.read())
            temp_file.flush()

            docs = await load_and_process_pdf_async(temp_file.name)
            embeddings = generate_embeddings(docs)
            index = create_faiss_index(embeddings)

            st.session_state.docs = docs
            st.session_state.faiss_index = index
            st.session_state.kernel = setup_kernel(index, docs)

            st.success("PDF loaded and system initialized successfully!")
        else:
            st.session_state.error = "Please upload a valid PDF."
    except Exception as e:
        st.session_state.error = str(e)
    finally:
        st.session_state.loading = False

# Main Streamlit app
st.set_page_config(page_title="Credit Card Assistant", layout="wide")
st.title("💳 Credit Card Assistant")
st.markdown("Ask about activating, deactivating cards, or general queries!")

# Sidebar for PDF Upload
with st.sidebar:
    st.header("Upload PDF Guide")
    uploaded_pdf = st.file_uploader("Upload your PDF document", type=["pdf"])
    if st.button("Load PDF"):
        if uploaded_pdf is not None:
            asyncio.run(handle_pdf_upload(uploaded_pdf))
        else:
            st.warning("Please upload a PDF file first!")

    st.divider()
    st.header("Quick Actions")
    card_number = st.text_input("Card Number", placeholder="XXXX-XXXX-XXXX-XXXX")

    if st.button("Activate Card"):
        if card_number:
            user_message = f"Activate card {card_number}"
            st.session_state.chat_history.append(("user", user_message))
        else:
            st.warning("Please enter a card number.")

    if st.button("Deactivate Card"):
        if card_number:
            user_message = f"Deactivate card {card_number}"
            st.session_state.chat_history.append(("user", user_message))
        else:
            st.warning("Please enter a card number.")

# Main chat area
if st.session_state.error:
    st.error(st.session_state.error)

user_input = st.text_input("Ask a question", key="input")

# Handle user sending a message
if st.button("Send"):
    if user_input:
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.user_just_sent = True  # track sending

# Process user's latest message if needed
if st.session_state.get("user_just_sent", False):
    if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
        with st.spinner("Thinking..."):
            latest_message = st.session_state.chat_history[-1][1]
            response = asyncio.run(process_message(latest_message))
            st.session_state.chat_history.append(("assistant", response))
    st.session_state.user_just_sent = False  # reset

# Display full chat history
for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").markdown(message)
    else:
        st.chat_message("assistant").markdown(message)

# Processing new messages
if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "user":
    with st.spinner("Thinking..."):
        user_message = st.session_state.chat_history[-1][1] if isinstance(st.session_state.chat_history[-1], tuple) else st.session_state.chat_history[-1]
        response = asyncio.run(process_message(user_message))
        st.session_state.chat_history.append(("assistant", response))