import streamlit as st
import torch
import pickle
from config import CFG
from model import NamedEntityRecog
from utils import  build_pretrain_embedding 
import logging
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset import NERDataset
from torchtext.data import get_tokenizer
import re
import pandas as pd
import json
import streamlit_ext as ste

st.title("Named Entity Recognition")

def convert_df(df:pd.DataFrame):
     return df.to_csv(index=False).encode('utf-8')


def convert_json(df:pd.DataFrame):
    result = df.to_json(orient="index")
    parsed = json.loads(result)
    json_string = json.dumps(parsed)
    #st.json(json_string, expanded=True)
    return json_string

@st.cache(allow_output_mutation=True)
def load_model():
    seed_num = 42
    random.seed(seed_num)
    torch.manual_seed(seed_num)
    np.random.seed(seed_num)
    with open(CFG.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)

    word2idx = lookup['word2idx']
    entity2idx = lookup['entity2idx']
    idx2entity = {idx: ent for ent, idx in entity2idx.items()}
    
    num_vocabs = len(word2idx)
    num_entities = len(entity2idx)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrain_word_embedding = build_pretrain_embedding(CFG.pretrained_embed_path, word2idx, CFG.word_embed_dim)
    
    model = NamedEntityRecog(num_vocabs , CFG.word_embed_dim , CFG.lstm_hidden_dim,num_entities , pretrain_embed=pretrain_word_embedding).to(device)
    if device =='cuda':
        model.load_state_dict(torch.load(CFG.trained_best_model))
    else:
        model.load_state_dict(torch.load(CFG.trained_best_model , map_location=torch.device('cpu')))

    return word2idx , entity2idx ,model , device , idx2entity

def predict(text , word2idx , entity2idx , model ,device ,idx2entity):
    tokenizer = get_tokenizer('basic_english')
    tokens = tokenizer(text)
    processed_t =[]
    for w in tokens:
        w= w.split('-')
        processed_t.extend(w)
    processed_t = np.array(processed_t)
    processed_t = np.expand_dims(processed_t, axis=0)
    test_dataset = NERDataset(word2idx, entity2idx, processed_t,inference=True)

    test_dataloader = DataLoader(test_dataset,batch_size=CFG.batch_size) 
    model.eval()

    for i , batch in enumerate(test_dataloader):
            sents = batch['text'].to(device)
            preds = model.inferences(sents) 
            pred_id2entity = [idx2entity[id_ent] for id_ent in preds[0]]
            print(pred_id2entity)

    df =pd.DataFrame()
    df['words'] = np.squeeze(processed_t)
    df['entity'] = pred_id2entity
    df.to_csv(CFG.result_csv,index=False)

    return df

with st.form(key='my_form'):

    x1 = st.text_input(label='Enter a sentence:', max_chars=250)
    print(x1)
    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        if re.sub('\s+','',x1)=='':
            st.error('Please enter a non-empty sentence.')

        elif re.match(r'\A\s*\w+\s*\Z', x1):
            st.error("Please enter a sentence with at least one word")
        
        else:
            st.markdown("### Tagged Sentence")
            st.header("")

            word2idx , entity2idx ,model , device , idx2entity = load_model()

            results = predict(x1 ,word2idx , entity2idx ,model , device , idx2entity)

            cs, c1, c2, c3, cLast = st.columns([0.75, 1.5, 1.5, 1.5, 0.75])

            with c1:
                csvbutton = ste.download_button(label="Download .csv", data=convert_df(results), file_name= "results.csv", mime='text/csv')
            with c2:
                textbutton = ste.download_button(label="Download .txt", data=convert_df(results), file_name= "results.text", mime='text/plain')
            with c3:
                jsonbutton = ste.download_button(label="Download .json", data=convert_json(results), file_name= "results.json", mime='application/json')

            st.header("")

            c1, c2, c3 = st.columns([1, 3, 1])

            with c2:

                st.table(results)

st.header("")
st.header("")
st.header("")