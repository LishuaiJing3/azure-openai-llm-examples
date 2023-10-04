# %%
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from src.utils.azOpenAILLM import AzureOpenAIChatModel


import os
import dotenv

dotenv.load_dotenv()

# %%
# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_3")

# Initialize LLM. Note that you need to use the chat model here to aviod output issues with the agent in the following.
llm_init = AzureOpenAIChatModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-35-turbo")
llm = llm_init.get_llm()

# %%
iris_df = pd.read_csv(
    "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
)

# %%
pd_df_agent = create_pandas_dataframe_agent(llm, iris_df, verbose=True)

pd_df_agent.run(
    "Analyze this data, tell me any interesting trends. Make some pretty plots."
)
# %%
# Can we get a LLM model to do some ML!?
pd_df_agent.run(
    "Train a classifer for the type of iris using the most important features. Show me the what variables are most influential to this model"
)
# %%
