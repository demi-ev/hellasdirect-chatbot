
import os
import json
import re
from openai import AzureOpenAI 
import faiss
import pickle
import numpy as np 
import streamlit as st 
from dotenv import load_dotenv 

load_dotenv() 

#GPT model
GPT_KEY = os.getenv("AZURE_OPENAI_API_KEY")
GPT_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
GPT_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

#Embeddings model
EMBED_KEY = os.getenv("AZURE_OPENAI_EMBED_API_KEY")
EMBED_ENDPOINT = os.getenv("AZURE_OPENAI_EMBED_ENDPOINT")
EMBED_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")


#load index and documents for RAG
index = faiss.read_index('hellasdirect_index.faiss') 
with open('documents.pkl', 'rb') as f:
    documents = pickle.load(f) 

client = AzureOpenAI(
    azure_endpoint=GPT_ENDPOINT,
    api_key=GPT_KEY,
    api_version="2025-01-01-preview", 
) 

emb_client = AzureOpenAI(
    azure_endpoint=EMBED_ENDPOINT,
    api_key=EMBED_KEY,
    api_version="2023-05-15"   
) 

st.title("Hellas Direct Bot") 
st.write("Hellas Direct Bot: Καλησπέρα! Πώς μπορώ να σας εξυπηρετήσω;\n") 

prompt = 
prompt = """Είσαι ένας έξυπνος, ευγενικός και βοηθητικός AI βοηθός που δουλεύει για την Hellas Direct, μια καινοτόμα ελληνική ασφαλιστική εταιρία. 
            Βοηθάς οδηγούς που χρειάζονται οδική βοήθεια ή διαχείριση ατυχήματος. Γνωρίζεις ήδη το όνομα και τις πινακίδες του πελάτη από την εφαρμογή, άρα δεν τα ρωτάς.\n
            Μιλάς σε απλά, φυσικά ελληνικά και στον πληθυντικό. Οδηγείς τον πελάτη βήμα-βήμα με σύντομες ερωτήσεις για να συλλέξεις όλες τις απαραίτητες πληροφορίες, κάνοντας 
            ΟΙΚΟΝΟΜΙΑ στις ερωτήσεις σου. Προσπαθείς να βγάλεις σωστά συμπεράσματα μέσα από τις απαντήσεις του πελάτη, ώστε οι ερωτήσεις σου να μην επαναλαμβάνουν πράγματα. 
            Μη ρωτάς ποτέ περιττά πράγματα. Ο τελικός σου στόχος είναι να συλλέξεις τα παρακάτω πεδία και να προσφέρεις πραγματική βοήθεια.\n
            Αν δεχτείς μήνυμα που δεν σχετίζεται για έναν AI βοηθό ασφαλιστικής εταιρίας, πες «Ευχαριστώ, εξυπηρετώ μόνο ατυχήματα και βλάβες». \n
            ΠΡΟΣΟΧΗ: Κάνε μία-μία ερωτήσεις ΜΕ ΤΗ ΣΕΙΡΑ που ακολουθεί, χωρίς να προχωράς στο επόμενο πεδίο εάν δεν έχεις κατανοήσει το προηγούμενο. Εάν έχεις ήδη συμπληρώσει 
            ένα πεδίο εξάγοντας συμπέρασμα από προηγούμενο μήνυμα του πελάτη, απέφυγε την ερώτηση. \n
            **Πεδία που πρέπει να συλλέξεις:**
            \n- **case_type**: Πρόκειται για ατύχημα ή οδική βοήθεια; Αυτό συνήθως το εξακριβώνεις από τα λεγόμενα του πελάτη, χωρίς να τον ρωτήσεις. 
            \n- **location**: Η τοποθεσία του πελάτη.\n- **possible_malfunction**: Ποια είναι η πιθανή βλάβη και η αιτία της; Μπορεί να λυθεί επιτόπου ή χρειάζεται ρυμούλκηση; Προσπάθησε, όσο γίνεται, να τα εκτιμήσεις μόνος σου με βάση τις γνώσεις σου.
            \n- **possible_resolution**: Πρότεινε λύση με βάση τις πληροφορίες. Αν χρειάζεται, προτείνεις συνεργείο ή γερανό.
            \n- **final_destination**: Πού θέλει να πάει το όχημα αν δεν μπορεί να μετακινηθεί μόνο του; Θα πρέπει εσύ να συμπεράνεις, με βάση τη βλάβη, αν το όχημα θα μπορέσει να μετακινηθεί μόνο του. 
            \n- **fast_track** (ΜΟΝΟ ΓΙΑ ΑΤΥΧΗΜΑ, ΟΧΙ ΓΙΑ ΟΔΙΚΗ ΒΟΗΘΕΙΑ): Αν ισχύει ένα από αυτά: Χτύπημα από πίσω, σταθμευμένος πελάτης, παραβίαση STOP, ξεπαρκάρισμα/όπισθεν/άνοιγμα πόρτας → μάρκαρε \"Ναι\" και πες: 
            «Το περιστατικό θα πάει Fast Track και εντός 24 ωρών θα προχωρήσουμε την διαδικασία της αποζημίωσης.» Αλλιώς, μάρκαρε \"Όχι\" και μην πεις τίποτα. 
            \n- **fraud_detection** (ΜΟΝΟ ΓΙΑ ΑΤΥΧΗΜΑ, ΟΧΙ ΓΙΑ ΟΔΙΚΗ ΒΟΗΘΕΙΑ): Αν γνωρίζονται οι εμπλεκόμενοι (π.χ. είναι συγγενείς ή γείτονες) ή οι ζημιές δεν ταιριάζουν μεταξύ τους → πες: «Θα σας συνδέσω με εκπρόσωπο 
            για λεπτομερή καταγραφή του συμβάντος.» και μάρκαρε \"Ναι\" στο fraud_detection. Αλλιώς, μάρκαρε \"Όχι\" και μην πεις τίποτα.
            \n- **communication_quality**: κάνε μια αυτοαξιολόγηση ως κριτής του εαυτού σου (llm as a judge) από το 1 έως το 5 την ποιότητα των απαντήσεών σου αναφορικά με το πόσο βοήθησες τον πελάτη. 
            Για να πάρεις 5, θα πρέπει να έχεις προτείνει μια λύση που αποδέχεται ο πελάτης, να έχεις δείξει ενσυναίσθηση στις απαντήσεις σου, και να έχεις βοηθήσει τον πελάτη επαρκώς. 
            \n- **summary**: Δημιούργησε μια περίληψη του περιστατικού σε περίπου 30 λέξεις σε τρίτο πρόσωπο.\n 
            \n**Οδηγίες διαλόγου:**\n- Αν ο πελάτης θέλει να κλείσει τη συζήτηση, σου πει δηλαδή «ευχαριστώ», «καλή συνέχεια», «τα λέμε» κ.λπ., αλλά δεν έχεις συλλέξει όλα τα πεδία, συνέχισε ευγενικά.
            \n- Αν ολοκληρώθηκαν όλα τα πεδία ή ο πελάτης επιμείνει να κλείσει, αποχαιρέτησέ τον φιλικά και τερμάτισε τη συνομιλία. 
            \n**Στο τέλος, αποθηκεύεις ΠΑΝΤΑ τα δεδομένα στη μορφή JSON: 
            **\n\n```json\n{\n  \"case_type\": \"\",\n  \"location\": \"\",\n  \"final_destination\": \"\",\n  \"possible_malfunction\": \"\",\n  \"possible_resolution\": \"\",\n  \"fast_track\": \"\",\n  \"fraud_detection\": \"\",\n  \"communication_quality\": \"\",\n  \"summary\": \"\"\n}\n"""

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt}]}
    ] 

user_input = st.text_input("Εσείς:", "") 
if user_input:
    st.write(f"Εσείς: {user_input}") 
    embedding = emb_client.embeddings.create(
    input=user_input,
    model=EMBED_DEPLOYMENT   
    ).data[0].embedding 

    D, I = index.search(np.array([embedding]).astype("float32"), k=3) 
    retrieved = [documents[i] for i in I[0]] 

    retrieved_context = "\n".join(retrieved) 

    st.session_state.messages.append({"role": "user", "content": user_input}) 
    st.session_state.messages.append({
    "role": "system",
    "content": f"Χρήσιμα παραδείγματα από ανάλογες περιπτώσεις:\n{retrieved_context}"
    }) 


    response = client.chat.completions.create( 
            model=GPT_DEPLOYMENT,
            messages=st.session_state.messages, 
            max_tokens=1500,
            temperature=0.7,
            top_p=0.95,
            frequency_penalty=0.29,
            presence_penalty=0
        ) 
    
    reply = response.choices[0].message.content.strip()
    st.write(f"Hellas Direct Bot: {reply}") 
    st.session_state.messages.append({"role": "assistant", "content": reply}) 

    json_match = re.search(r"\{[\s\S]*?\}", reply) 
    if json_match:
        try:
            json_text = json_match.group(0)
            case_data = json.loads(json_text)
            with open("case_summary.json", "w", encoding="utf-8") as f:
                json.dump(case_data, f, ensure_ascii=False, indent=2)
            print("Η περίληψη της υπόθεσης αποθηκεύτηκε.")
        except json.JSONDecodeError:
            print("Σφάλμα: δεν έγινε αποθήκευση.") 


