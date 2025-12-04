
import json
from openai import AzureOpenAI 
import faiss 
import numpy as np
import pickle
import os

EMBED_KEY = os.getenv("AZURE_OPENAI_EMBED_API_KEY")
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT") 
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

with open('knowledge_base_rag.json', 'r', encoding='utf8') as f:
    raw_data = json.load(f) 

documents = [entry["content"] for entry in raw_data] 

client = AzureOpenAI(
    azure_endpoint=EMBED_ENDPOINT,
    api_key=EMBED_KEY,
    api_version="2023-05-15"  
) 

embeddings = [] #turn docs into vectors (embeddings)
for text in documents:
    response = client.embeddings.create( 
        input=text,
        model=EMBED_DEPLOYMENT
    )
    embeddings.append(response.data[0].embedding) 

dimension = len(embeddings[0]) #number of dimensions in each embedding vector
index = faiss.IndexFlatL2(dimension) #L2 distance
index.add(np.array(embeddings).astype("float32")) #add doc embeddings to FAISS index

faiss.write_index(index, 'hellasdirect_index.faiss') #save index
with open('documents.pkl', 'wb') as f: #save docs
    pickle.dump(documents, f) 

