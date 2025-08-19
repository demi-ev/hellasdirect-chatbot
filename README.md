
# Hellas Direct Chatbot using RAG and Azure

This project was undertaken following a challenge in UniAI's Makeathon held on 2025 in Athens. One of the sponsors, Hellas Direct, a Greek insurance company, needed a chatbot for customer support due to its shortage of human agents.

The chatbot should be able to recognize if there is a case of road assistance or accident care, as well as understand the customer's problem and provide a satisfactory solution according to the company's regulations.

## Overview of the architecture

- Used an Azure OpenAI GPT deployment for the chatbot's responses
- Used an Azure OpenAI embeddings deployment + a .json file of mock client/human agent calls for RAG
- Used FAISS to create a vector database for retrieval
- Implemented a simple frontend component with Streamlit

## Requirements

To run this project, an Azure subscription is needed. You will need:

- An Azure OpenAI resource
- A GPT deployment
- An embeddings deployment
- API keys for your Azure OpenAI resource 

Your .env file should include:

- AZURE_OPENAI_API_KEY=
- AZURE_OPENAI_ENDPOINT=
- AZURE_OPENAI_DEPLOYMENT=

- AZURE_OPENAI_EMBED_API_KEY=
- AZURE_OPENAI_EMBED_ENDPOINT=
- AZURE_OPENAI_EMBED_DEPLOYMENT= (replace in code)

Running locally via Python:

- clone the repo
- pip install -r requirements.txt
- create .env file with Azure info
- streamlit run chatbot-streamlit.py 

## Example Use on Streamlit


![Example UI](assets/chbt-display.png)

## Limitations

- There is an issue with the variable 'AZURE_OPENAI_EMBED_DEPLOYMENT', and thus the model variable has to be replaced manually. This corresponds to "chbt-embedding" in the present code.
- The chatbot was created mostly for practice purposes, and so there is ample room for improvement as to how it handles edge cases and complex queries.
- No significant evaluation was conducted, although this is a priority for future improvements.




