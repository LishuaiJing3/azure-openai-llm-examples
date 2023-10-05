## Disclaimer

This repo is mainly used for sharing code and experiences. It is not production-ready code even through many coding standards and good practices have been applied. Take this as an inspiration and hope you can find some of it be useful for your projects. 

There are many parameters that you can tune to control the LLM, I will leave it to your effort to find the best config for your projects.

## Introduction

There are many examples on how to use Open AI services with different tools. The resources on using Azure OpenAI service is comparably small. This repo provides some boilder code to show how to integrate Azure OpenAI service with

- vectordb
- huggingface
- langchain 
- databricks SQL warehouse to chat with your existing datasets. 

It expriments general purpose and chat version Azure OpenAI GPT-3 and GPT-4 models.  

For vector DB, Chroma db is used and huggingface embeddings are used for creating vectors for documents.   

## Prerequisites

- [Azure openAI service and deployments](https://learn.microsoft.com/en-us/azure/ai-services/openai/)
- Hugging face account (Optional, only if some of the boiler code requires it)
- Google search API (Optional, only if some of the boiler code requires it) 
- Azure databricks SQL warehouses (Optional, only if some of the boiler code requires it)

## Observations
### Databricks 
When using gpt-3-turbo general purpose models, It seems the output is quite difficult to control and many times it results in OutputParserException. This is due to that AzureChatOpenAI is needed, if other type of model (e.g. AzureOpenAI) is used, then it will cause this issue. 
### Pkg dependencies
It can cause many issues is the packages are not compatible. Double check pkg dependencies.  

## Issues:
- Cannot connect to Azure OpenAI service. This can happen when firwarewalls are enabled. try a different network if you run into this issue.  

## References
- [Azure OpenAI service](https://learn.microsoft.com/en-us/azure/ai-services/openai/quickstart?tabs=command-line&pivots=programming-language-studio)
- [Langchain Docs](https://python.langchain.com/docs/integrations/providers/databricks)
- [HuggingFace](https://huggingface.co/docs)
- [ChromaDB](https://docs.trychroma.com/)