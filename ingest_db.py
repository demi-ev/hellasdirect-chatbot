
import json
from openai import AzureOpenAI 
import faiss 
import numpy as np
import pickle

with open('knowledge_base_rag.json', 'r', encoding='utf8') as f:
    raw_data = json.load(f) 

documents = [entry["content"] for entry in raw_data] 

client = AzureOpenAI(
    azure_endpoint="https://openai-emb-chbt.openai.azure.com/",
    api_key="2QmwUXQgrBHG473QojgKcopfhVxUArSdQ5kqNxaD2W3JBjxIsKxcJQQJ99BGACfhMk5XJ3w3AAABACOGWheg",
    api_version="2023-05-15"  
) 

embeddings = [] #turn docs into vectors (embeddings)
for text in documents:
    response = client.embeddings.create( 
        input=text,
        model="chbt-embedding"
    )
    embeddings.append(response.data[0].embedding) 

dimension = len(embeddings[0]) #number of dimensions in each embedding vector
index = faiss.IndexFlatL2(dimension) #L2 distance
index.add(np.array(embeddings).astype("float32")) #add doc embeddings to FAISS index

faiss.write_index(index, 'hellasdirect_index.faiss') #save index
with open('documents.pkl', 'wb') as f: #save docs
    pickle.dump(documents, f) 
