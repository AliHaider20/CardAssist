💳 CardAssist: Multi-Agent Credit Card Assistant using Semantic Kernel

CardAssist is a multi-agent AI assistant that leverages Semantic Kernel, OpenAI, and Retrieval-Augmented Generation (RAG) to perform intelligent credit card management tasks like activation, deactivation, and natural language Q&A from a PDF-based user guide.
It’s designed to be easily extensible, enabling conversational, grounded assistance for end users.

⸻

🚀 Features
	•	✅ Activate/Deactivate Credit Cards using natural language
	•	🔍 RAG-based QA using embeddings from PDF user guides
	•	⚙️ Powered by Semantic Kernel Plugins & Azure OpenAI
	•	⚡ Vector Search via FAISS
	•	📄 PDF ingestion with Langchain + Sentence Transformers
	•	🧠 Context-aware system prompt for consistent, helpful replies

⸻

📂 Project Structure

CardAssist/
├── global_card_access_user_guide.pdf   # PDF used for RAG
├── main.ipynb                          # Jupyter Notebook for full pipeline
├── .env                                # Environment variables (not committed)
└── README.md



⸻

🛠️ Setup Instructions
	1.	Clone this repository

git clone https://github.com/yourusername/CardAssist.git
cd CardAssist

	2.	Create and activate a virtual environment

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

	3.	Install dependencies

pip install -r requirements.txt

	4.	Set up your .env file

AI_FOUNDRY_MODEL_API=your_openai_or_foundry_key
LLM_ENDPOINT=https://your-openai-endpoint
AI_SEARCH_API=your_search_api_key

	5.	Run the Notebook
Open main.ipynb in JupyterLab or VSCode and follow through the cells.

⸻

🔧 Core Components

🔹 Embedding & Indexing
	•	Uses sentence-transformers/all-MiniLM-L6-v2 to embed PDF content.
	•	Chunks indexed with FAISS for fast similarity search.

🔹 Semantic Kernel Plugins

@kernel_function
async def activate_card(card_number: str) -> str

Custom functions for activating/deactivating cards and handling RAG-based queries.

🔹 Azure OpenAI Integration
	•	Uses AzureChatCompletion with GPT-4o (gpt-4o-mini) via Semantic Kernel.

⸻

💬 Sample Interactions

🔧 Activate Card

“Can you activate my credit card 1234-5678-9012-3456?”

Response: Credit card 1234-5678-9012-3456 has been activated.

❌ Deactivate Card

“Please deactivate my card ending in 3456”

Response: Credit card ending in 3456 has been successfully deactivated.

🧠 Ask a Question (RAG)

“Steps for First-time Registration for Corporate Accounts”

Response: To register as a new user for a corporate account, follow these steps...



⸻

📌 Dependencies
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

📈 Future Enhancements
	•	✅ Add Gradio/Streamlit-based Chat UI
	•	🔐 Mask sensitive card numbers in output
	•	🧩 Add more card-related management functions (limit changes, billing FAQs)
	•	☁️ Deploy as Azure Web App or API endpoint

⸻

👨‍💻 Author

Haider – LinkedIn | Portfolio

⸻

Let me know if you’d like a requirements.txt or badge support (e.g. build, Python version, license, etc.) too!
