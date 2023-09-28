# %%
from src.utils.AzOpenaiLLM import AzureOpenAIChatModel
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.utilities import SQLDatabase

import os
import dotenv

dotenv.load_dotenv()
# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
# %%
# Initialize LLM: Note that chat model is used here rather than general model
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_4")
llm_init = AzureOpenAIChatModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-4")
llm = llm_init.get_llm()
# %%
# Connecting to Databricks with SQLDatabase wrapper


db_host = os.getenv("DATABRICKS_HOST")
db_token = os.getenv("DATABRICKS_TOKEN")
db_warehouse_id = os.getenv("DATABRICKS_SQL_WAREHOUSE_ID")

db_catalog = "samples"
db_schema = "nyctaxi"

# db_cluster_id
db = SQLDatabase.from_databricks(
    catalog=db_catalog,
    schema=db_schema,
    host=db_host,
    api_token=db_token,
    warehouse_id=db_warehouse_id,
)

# %%

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# Chat specific model is needed otherwise the reuslt has issues with parsing errors
agent = create_sql_agent(llm=llm, toolkit=toolkit, top_k=2, verbose=True)
question = "What is the largest fare amount?"
results = agent.run(question)
print(results)
