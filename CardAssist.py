import asyncio
import os
import gradio as gr
import numpy as np
import faiss
from tqdm import tqdm
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import AzureChatPromptExecutionSettings
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions import kernel_function
from typing import Annotated

import nest_asyncio
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Initialize embedding model
embedding_service = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# PDF processing functions
def load_and_process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=30)
    md_docs = text_splitter.split_documents(documents)
    md_docs = [doc.page_content for doc in md_docs]
    return md_docs

# Generate document embeddings
def generate_all_embeddings(docs):
    embeddings = []
    for doc in tqdm(docs, desc="Generating embeddings"):
        embedding = embedding_service.encode([doc])
        embeddings.append(embedding[0])
    return embeddings

# Create FAISS index
def create_faiss_index(embeddings):
    embedding_matrix = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embedding_matrix.shape[1])
    index.add(embedding_matrix)
    return index

# Credit Card Plugin for Semantic Kernel
class CreditCardPlugin:
    def __init__(self, faiss_index, docs):
        self.index = faiss_index
        self.docs = docs
        
    @kernel_function(
        description="Deactivate a credit card; returns a confirmation message"
    )
    async def deactivate_card(self, card_number: Annotated[str, "The credit card number to deactivate"]) -> str:
        return f"Credit card {card_number} has been deactivated."

    @kernel_function(
        description="Activate a credit card; returns a confirmation message"
    )
    async def activate_card(self, card_number: Annotated[str, "The credit card number to activate"]) -> str:
        return f"Credit card {card_number} has been activated."
    
    @kernel_function(
        description="Get card information and other general information about the card and account management"
    )
    async def rag_query(self, query: Annotated[str, "The user query for RAG (Retrieval-Augmented Generation)"]):
        query_embedding = embedding_service.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k = 5  # Get top 5 most relevant chunks
        distances, indices = self.index.search(query_embedding, k)
        relevant_chunks = [self.docs[i] for i in indices[0]]

        context = "\n".join(relevant_chunks)
        augmented_prompt = f"Based on the following information from the Global Card Access user guide:\n\n{context}\n\nUser Query: {query}\n\nPlease provide a helpful and accurate response:"
        
        return augmented_prompt

# Setup Semantic Kernel
def setup_kernel(faiss_index, docs):
    kernel = Kernel()

    # Setup Azure OpenAI
    model_id = "gpt-4o-mini"
    llm_endpoint = os.getenv("LLM_ENDPOINT")
    api_key = os.getenv("AI_FOUNDRY_MODEL_API")

    # Add the Azure OpenAI chat completion service
    kernel.add_service(
        AzureChatCompletion(
            deployment_name=model_id,
            endpoint=llm_endpoint,
            api_key=api_key
        )
    )

    # Add the credit card plugin
    kernel.add_plugin(
        CreditCardPlugin(faiss_index, docs),
        plugin_name="CreditCard",
    )
    
    return kernel

# Chat processing function
async def process_message(kernel, message, history):
    arguments = KernelArguments(
        settings=AzureChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto(),
            top_p=0.9,
            temperature=0,
        )
    )
    
    response = await kernel.invoke_prompt(message, arguments=arguments)
    return response.value[0].content

# Initialize the system
def initialize_system(pdf_path):
    # Load and process the PDF
    docs = load_and_process_pdf(pdf_path)
    
    # Generate embeddings
    embeddings = generate_all_embeddings(docs)
    
    # Create FAISS index
    faiss_index = create_faiss_index(embeddings)
    
    # Setup the kernel
    kernel = setup_kernel(faiss_index, docs)
    
    return kernel

# Gradio interface
def create_gradio_interface():
    with gr.Blocks(title="Credit Card Assistant", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Credit Card Assistant")
        gr.Markdown("Ask questions about your credit card, activate or deactivate cards, and get information from the Global Card Access user guide.")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500)
                msg = gr.Textbox(label="Type your message here", placeholder="How do I activate my credit card?")
                send_btn = gr.Button("Send")
            
            with gr.Column(scale=1):
                with gr.Accordion("PDF Upload", open=False):
                    pdf_input = gr.File(label="Upload PDF Document", file_types=[".pdf"])
                    pdf_path_input = gr.Textbox(label="Or enter PDF path", placeholder="/path/to/your/file.pdf", 
                                               value="/Users/ha/Desktop/Projects/Wipro task/global_card_access_user_guide.pdf")
                    load_pdf_btn = gr.Button("Load PDF")
                
                with gr.Accordion("Quick Actions", open=True):
                    card_num = gr.Textbox(label="Credit Card Number", placeholder="XXXX-XXXX-XXXX-XXXX")
                    act_btn = gr.Button("Activate Card")
                    deact_btn = gr.Button("Deactivate Card")
                
                status = gr.Textbox(label="Status", value="System initialized with default PDF")

        # Initialize system with default PDF path
        kernel = initialize_system("/Users/ha/Desktop/Projects/Wipro task/global_card_access_user_guide.pdf")
        
        # Function to handle chat interaction
        def respond(message, chat_history):
            chat_history.append((message, ""))
            return "", chat_history
        
        async def bot_response(chat_history):
            message = chat_history[-1][0]
            response = await process_message(kernel, message, chat_history[:-1])
            chat_history[-1] = (message, response)
            return chat_history
        
        # Function to handle PDF loading
        def load_pdf(file, path):
            nonlocal kernel
            try:
                if file is not None:
                    pdf_path = file.name
                elif path and os.path.exists(path):
                    pdf_path = path
                else:
                    return "Error: No valid PDF provided"
                
                kernel = initialize_system(pdf_path)
                return f"Successfully loaded PDF from {pdf_path}"
            except Exception as e:
                return f"Error loading PDF: {str(e)}"
            
        # Functions for quick actions
        async def activate_card_action(number, chat_history):
            if not number:
                chat_history.append(("Activate my card", "Please provide a card number."))
                return chat_history
            
            chat_history.append((f"Activate card {number}", ""))
            response = await process_message(kernel, f"Please activate my credit card with number {number}", chat_history[:-1])
            chat_history[-1] = (f"Activate card {number}", response)
            return chat_history
            
        async def deactivate_card_action(number, chat_history):
            if not number:
                chat_history.append(("Deactivate my card", "Please provide a card number."))
                return chat_history
                
            chat_history.append((f"Deactivate card {number}", ""))
            response = await process_message(kernel, f"Please deactivate my credit card with number {number}", chat_history[:-1])
            chat_history[-1] = (f"Deactivate card {number}", response)
            return chat_history
            
        # Connect components
        send_btn.click(respond, [msg, chatbot], [msg, chatbot]).then(
            lambda x: asyncio.run(bot_response(x)), [chatbot], [chatbot]
        )
        
        msg.submit(respond, [msg, chatbot], [msg, chatbot]).then(
            lambda x: asyncio.run(bot_response(x)), [chatbot], [chatbot]
        )
        
        load_pdf_btn.click(load_pdf, [pdf_input, pdf_path_input], [status])
        
        act_btn.click(
            lambda x, y: asyncio.run(activate_card_action(x, y)), 
            [card_num, chatbot], 
            [chatbot]
        )
        
        deact_btn.click(
            lambda x, y: asyncio.run(deactivate_card_action(x, y)), 
            [card_num, chatbot], 
            [chatbot]
        )
        
    return app

# Launch the Gradio app
if __name__ == "__main__":
    app = create_gradio_interface()
    app.launch(share=True)