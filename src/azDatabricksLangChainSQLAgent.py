# %%
from src.utils.azOpenAILLM import AzureOpenAIChatModel

# import databricksConn
from src.utils.databrickConn import databricksConn
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit

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

db_init = databricksConn(db_host, db_token, 
                         db_warehouse_id, 
                         db_catalog, db_schema)
db = db_init.get_db()

# %%

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
# Chat specific model is needed otherwise the reuslt
# has issues with parsing errors
agent = create_sql_agent(llm=llm, toolkit=toolkit, top_k=2, verbose=True)
question = "What is the largest fare amount?"
results = agent.run(question)
print(results)

# %%
