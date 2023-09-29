# %%
import os

from src.utils.AzOpenaiLLM import AzureOpenAIModel
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import VectorDBQA
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import dotenv

dotenv.load_dotenv()
# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_3")
# %%
llm_init = AzureOpenAIModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-35-turbo")
llm = llm_init.get_llm()
llm("Tell me a joke")
# %%
doc_loader = PyPDFLoader("../data/genAIEco.pdf")
pages = doc_loader.load_and_split()
# First we split the data into manageable chunks to store as vectors. There isn't an exact way to do this, more chunks means more detailed context, but will increase the size of our vectorstore.
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=10)
texts = text_splitter.split_documents(pages)
# Now we'll create embeddings for our document so we can store it in a vector store and feed the data into an LLM. 
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=model_name)
# Finally we make our Index using chromadb and the embeddings LLM
chromadb_index = Chroma.from_documents(texts, embeddings)

# %%
# build qa model
qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=chromadb_index)
# %%
query = "How much money can generative AI contribute?"
qa.run(query)
query = "What is the GDP for United kingdom?"
qa.run(query)
