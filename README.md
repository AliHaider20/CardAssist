# 💳 CardAssist: Multi-Agent Credit Card Assistant using Semantic Kernel

**CardAssist** is a multi-agent AI assistant that leverages **Semantic Kernel**, **OpenAI**, and **Retrieval-Augmented Generation (RAG)** to perform intelligent credit card management tasks such as activation, deactivation, and natural language Q&A from a PDF-based user guide.  
Designed to be modular and extensible, it enables conversational, grounded assistance for end users.

---

### 🚀 Features

- ✅ Natural Language Activation/Deactivation of credit cards  
- 🔍 RAG-powered Q&A using semantic embeddings from PDF guides  
- ⚙️ Semantic Kernel Plugins orchestrating skillful agent responses  
- ⚡ FAISS-based Vector Search for fast and efficient retrieval  
- 📄 PDF Ingestion via Langchain + Sentence Transformers  
- 🧠 Context-Aware Prompting to maintain coherent and helpful replies  

---


### 📁 Project Structure

```text
CardAssist/
├── global_card_access_user_guide.pdf   
├── CardAssist.ipynb                    # Main pipeline (Jupyter Notebook)
├── .env                                
└── README.md                           
```

---

### 🛠️ Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/yourusername/CardAssist.git
   cd CardAssist

	2.	Create and activate a virtual environment

python -m venv venv
source venv/bin/activate       # On macOS/Linux
# OR
venv\Scripts\activate          # On Windows


	3.	Install dependencies

pip install -r requirements.txt


	4.	Create a .env file and configure your API keys

AI_FOUNDRY_MODEL_API=your_openai_or_foundry_key
LLM_ENDPOINT=https://your-openai-endpoint
AI_SEARCH_API=your_search_api_key
EMBEDDING_ENDPOINT=your_embedding_endpoint


	5.	Run the notebook
Open CardAssist.ipynb in JupyterLab or VSCode and follow the execution cells.

⸻

🔧 Core Components

🔹 Embedding & Indexing
	•	Uses sentence-transformers/all-MiniLM-L6-v2 for PDF chunk embedding
	•	Embedded chunks are indexed using FAISS for fast similarity search

🔹 Semantic Kernel Plugins

@kernel_function
async def activate_card(card_number: str) -> str:

	•	Custom functions for activation, deactivation, and PDF-based queries

🔹 Azure OpenAI Integration
	•	Integrated with AzureChatCompletion using the gpt-4o-mini model via Semantic Kernel

⸻

💬 Sample Interactions

🔧 Activate Card

User: “Can you activate my credit card 1234-5678-9012-3456?”
💬 Response: Credit card 1234-5678-9012-3456 has been activated.

⸻

❌ Deactivate Card

User: “Please deactivate my card ending in 3456”
💬 Response: Credit card ending in 3456 has been successfully deactivated.

⸻

🧠 Ask a Question (RAG)

User: “Steps for First-time Registration for Corporate Accounts”
💬 Response: To register as a new user for a corporate account, follow these steps…

⸻

📦 Dependencies
	•	semantic-kernel
	•	langchain
	•	sentence-transformers
	•	faiss-cpu
	•	torch
	•	PyPDFLoader
	•	tqdm
	•	dotenv
	•	asyncio, nest_asyncio (for Jupyter async compatibility)

⸻

🔮 Future Enhancements
	•	🧠 Integrate Azure AI Search for advanced RAG indexing
	•	✅ 🧑‍💻 Add Gradio/Streamlit-based UI for user interaction
	•	🔐 Implement masking for sensitive card numbers
	•	📊 Add more card management capabilities (e.g., limit changes, billing FAQs)
	•	☁️ Deploy as an Azure Web App or RESTful API
