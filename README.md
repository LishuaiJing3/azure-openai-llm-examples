## Introduction

This repo provides some boilder code to show how to use azure openai service with vectordb, huggingface, langchain. It also shows how to connect to databricks SQL warehouse to chat with your existing datasets. 

For vector DB, Chroma db is used and huggingface embeddings are used for creating vectors for documents.   



## ToDos
- add rate limiting functions to control cost for azure open ai services. 

## Observations
### Databricks 
When using gpt-3-turbo models, It seems the output is quite difficult to control and many times it results in OutputParserException. Re-running the same some times get the right output. This is due to that AzureChatOpenAI is needed, if other type of model (e.g. AzureOpenAI) is used, then it will cause this issue. 
