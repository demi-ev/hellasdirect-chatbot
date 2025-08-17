
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


![Example UI](https://freeimage.host/i/Fm785AB](https://ibb.co/4nBscbL7)

