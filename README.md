## Disclaimer

This code repo is mainly used for sharing code and experiences. It is not production code even through many good coding standards and good practices have been applied. Take this as an inspiration and hope you can find some of it be useful for your projects. 

## Introduction

This repo provides some boilder code to show how to integrate Azure OpenAI service with - vectordb
- huggingface
- langchain 
- databricks SQL warehouse to chat with your existing datasets. 

It expriments general Azure OpenAI Chat GPT-3 and GPT-4 models.  

For vector DB, Chroma db is used and huggingface embeddings are used for creating vectors for documents.   

## Prerequisites

- (Azure openAI service and deployments)[https://learn.microsoft.com/en-us/azure/ai-services/openai/]
- Hugging face account (Optional, only if some of the boiler code requires it)
- Google search API (Optional, only if some of the boiler code requires it) 
- Azure databricks SQL warehouses (Optional, only if some of the boiler code requires it)

## Observations
### Databricks 
When using gpt-3-turbo models, It seems the output is quite difficult to control and many times it results in OutputParserException. Re-running the same some times get the right output. This is due to that AzureChatOpenAI is needed, if other type of model (e.g. AzureOpenAI) is used, then it will cause this issue. 
### Pkg dependencies
It can cause many issues is the packages are not compatible. Double check pkg dependencies.  

## ToDos
- add rate limiting functions to control cost for azure open ai services. 

## References
- (Azure OpenAI service)[https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-studio]
- (Langchain Docs)[https://python.langchain.com/docs/integrations/providers/databricks]
- (HuggingFace)[https://huggingface.co/docs]
- (ChromaDB)[https://docs.trychroma.com/]