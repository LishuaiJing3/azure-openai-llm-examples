#%%
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models.openai import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.embeddings import HuggingFaceEmbeddings
import os
import dotenv

dotenv.load_dotenv()
# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_3")
# %%
from src.utils.AzOpenaiLLM import AzureOpenAIModel

llm_init = AzureOpenAIModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-35-turbo")
llm = llm_init.get_llm()
# %%
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)

# Vectorstore
vectorstore = Chroma(embedding_function=embeddings,persist_directory="./chroma_db_oai")

#%%
# Search
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") 
search = GoogleSearchAPIWrapper()
# %%

# Initialize
web_research_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=llm, 
    search=search)
# %%
# Run
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.web_research").setLevel(logging.INFO)
from langchain.chains import RetrievalQAWithSourcesChain
#user_input = "How do LLM Powered Autonomous Agents work?"
user_input = "How much does a tesla model 3 cost in denmark?"

qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm,retriever=web_research_retriever)
result = qa_chain({"question": user_input})
result
# %%
