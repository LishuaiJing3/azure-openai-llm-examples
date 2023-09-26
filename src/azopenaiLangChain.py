# %%
import os
import dotenv

dotenv.load_dotenv()
# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_3")
# %%
from utils.AzOpenaiLLM import AzureOpenAIModel

llm_init = AzureOpenAIModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-35-turbo")
llm = llm_init.get_llm()
llm("Tell me a joke")
print(llm)

llm("Tell me a joke")

print(llm("Tell me a joke"))
print(llm)
# %%
from langchain import PromptTemplate, LLMChain

template = """
You are a support assistant. Answer the following questions to the best of your ability. Provide only answer or nothing else.
Qestion: {question}
Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(llm=llm, prompt=prompt)
# %%
question = "What does data scientist do everyday?"
print(llm_chain.run(question))
