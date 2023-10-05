# In this example, I test the Azure OpenAI LLM with a multi-stage reasoning task. We create two steps and evaluate the outcomes before setting up the sequence chain in the end.
from src.utils.azOpenAILLM import AzureOpenAIModel
from langchain import PromptTemplate
import numpy as np
from langchain.chains import SequentialChain
from langchain.chains import LLMChain
from better_profanity import profanity

import os
import dotenv

dotenv.load_dotenv()

# get base url, deployment name, and api key from .env file
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
DEPLOYMENT = os.getenv("DEPLOYMENT_GPT_3")
# %%
# Initialize LLM
llm_init = AzureOpenAIModel(BASE_URL, API_KEY, DEPLOYMENT, "gpt-35-turbo")
llm = llm_init.get_llm()
# %%
# make a template on how to prompt the LLM
customer_support_template = """
You are a customer supporter, you will respond to the following reviews with a {sentiment} response. 
Post:" {product_review}"
Comment: 
"""
# We use the PromptTemplate class to create an instance of our template that will use the prompt from above and store variables we will need to input when we make the prompt.
customer_support_prompt_template = PromptTemplate(
    input_variables=["sentiment", "product_review"],
    template=customer_support_template,
)

# Let us make a randomized sentiment
random_sentiment = "positive"
if np.random.rand() < 0.5:
    random_sentiment = "negative"
# create a product review
product_review = "I can't believe I have bought this product. The design is hard to see whether it is good or not but the color is ok."

# Let's create the prompt and print it out, this will be given to the LLM.
customer_support_prompt = customer_support_prompt_template.format(
    sentiment=random_sentiment, product_review=product_review
)
print(f"Customer suport prompt:{customer_support_prompt}")
# %%
customer_support_chain = LLMChain(
    llm=llm,
    prompt=customer_support_prompt_template,
    output_key="Customer_support_replied:",
    verbose=False,
)  # Now that we've chained the LLM and prompt, the output of the formatted 
#prompt will pass directly to the LLM.

# here we test the customer support chain response
customer_support_reply = customer_support_chain.run(
    {"sentiment": random_sentiment, "product_review": product_review}
)

# clean the review up using profanity. You can add some bad words here to test 
# what is happening:
cleaned_customer_support_reply = profanity.censor(customer_support_reply)
print(f"Moderator said:{cleaned_customer_support_reply}")
# %%
# 1 We will build the prompt template
# Our template for moderator will take the customer supporter reply and do some sentiment analysis.
mod_template = """
You are the moderator of an online forum. You are strict and will not tolerate any negative comments. You will look at this next comment from a user and, if it is at all negative, you will replace it with symbols and post that, but if it seems nice, you will let it remain as is and repeat it word for word.
Original comment: {customer_support_reply}
Edited comment:
"""
mod_prompt_template = PromptTemplate(
    input_variables=["customer_support_reply"],
    template=mod_template,
)
# Here we use the same model as beofre. You can change this to any model you want.
mod_llm = llm


mod_chain = LLMChain(
    llm=mod_llm, prompt=mod_prompt_template, verbose=False
)  # Now that we've chained the LLM and prompt, the output of the formatted prompt will pass directly to the LLM.

# To run our chain we use the .run() command and input our variables as a dict
mod_says = mod_chain.run({"customer_support_reply": customer_support_reply})
# Let's see what the response from the moderator is.
print(f"Moderator says: {mod_says}")
# %%
# The SequentialChain class takes in the chains we are linking together, as well as the input variables that will be added to the chain. These input variables can be used at any point in the chain, not just the start.
supporter_mod_chain = SequentialChain(
    chains=[customer_support_chain, mod_chain],
    input_variables=["sentiment", "product_review", "customer_support_reply"],
    verbose=True,
)

# We can now run the chain with our randomized sentiment, and the customer review!
supporter_mod_chain.run(
    {
        "sentiment": random_sentiment,
        "product_review": product_review,
        "customer_support_reply": customer_support_reply,
    }
)
# %%
# reference (https://github.com/databricks-academy/large-language-models/blob/published/LLM%2003%20-%20Multi-stage%20Reasoning/LLM%2003%20-%20Building%20LLM%20Chains.py)
