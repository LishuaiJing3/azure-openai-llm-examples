# %%
from langchain.agents import load_tools  # This will allow us to load tools we need
from langchain.agents import initialize_agent
from langchain.agents import (
    AgentType,
)  # We will be using the type: ZERO_SHOT_REACT_DESCRIPTION which is standard
from src.utils.azOpenAILLM import AzureOpenAIChatModel


import os
import dotenv

dotenv.load_dotenv()

# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_3")

SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
HUGGING_FACE_API = os.getenv("HUGGING_FACE_API")
# %%
# Initialize LLM. Note that you need to use the chat model here to aviod output issues with the agent in the following.
llm_init = AzureOpenAIChatModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-35-turbo")
llm = llm_init.get_llm()

# %% Search on wikipedia and serpapi to answer data science questions
tools = load_tools(["wikipedia", "serpapi"], llm=llm)
# We now initialize the agent
ds_agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# %%
ds_agent.run(
    "Find the most recent household income data on wikipedia (need to refer the right data source) and analyze the summary statistics. You need to summarize your findings."
)
# %%
