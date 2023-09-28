# %%
from langchain.vectorstores import Chroma
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from src.utils.AzOpenaiLLM import AzureOpenAIModel
from langchain.chains import RetrievalQAWithSourcesChain
import os
import dotenv
import logging

dotenv.load_dotenv()

logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)

# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_3")
# %%
# Initialize LLM
llm_init = AzureOpenAIModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-35-turbo")
llm = llm_init.get_llm()
# %%
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Vectorstore
vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db_oai")

# %%
# Search
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
search = GoogleSearchAPIWrapper()
# %%

# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore, llm=llm, search=search, 
    num_search_results=3
)
# %%
# Run
# user_input = "How do LLM Powered Autonomous Agents work?"
user_input = "How much does a tesla model 3 cost in denmark?"

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, retriever=web_research_retriever
)
result = qa_chain({"question": user_input})
result
# %%
