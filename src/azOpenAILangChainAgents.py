# %%
import os

from utils.AzOpenaiLLM import AzureOpenAIModel
from langchain.agents import load_tools, initialize_agent, AgentType

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
print(llm)
# %%
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
# %%
# initialize the agent
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, verbose=True)

agent_zero_shot = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
)

agent_zero_shot.run(
    "What is the age of the universe? What is the square root of it? stop process after the final answer"
)
