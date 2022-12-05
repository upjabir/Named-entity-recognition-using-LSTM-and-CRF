import torch
import pickle
from config import CFG
import time
from model import NamedEntityRecog
from utils import load_dataset , build_pretrain_embedding ,cal_f1score ,show_report
import logging
import random
import numpy as np
from torch.utils.data import DataLoader
from dataset import NERDataset
from torchtext.data import get_tokenizer
import argparse
import pandas as pd


seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

with open(CFG.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)

word2idx = lookup['word2idx']
entity2idx = lookup['entity2idx']
idx2entity = {idx: ent for ent, idx in entity2idx.items()}
o_entity = entity2idx['O']

num_vocabs = len(word2idx)
num_entities = len(entity2idx)

emb_begin = time.time()
pretrain_word_embedding = build_pretrain_embedding(CFG.pretrained_embed_path, word2idx, CFG.word_embed_dim)
emb_end = time.time()
emb_min = (emb_end - emb_begin) % 3600 // 60
logger.info('build pretrain embed cost {}m with shape {}'.format(emb_min,pretrain_word_embedding.shape))


def main(text):

    if text is None:
        test_data = load_dataset(CFG.test_data)
        test_dataset = NERDataset(word2idx, entity2idx, test_data)
    else:
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Device : {}".format(device))

    model = NamedEntityRecog(num_vocabs , CFG.word_embed_dim , CFG.lstm_hidden_dim,num_entities , pretrain_embed=pretrain_word_embedding).to(device)
    if device =='cuda':
        model.load_state_dict(torch.load(CFG.trained_best_model))
    else:
        model.load_state_dict(torch.load(CFG.trained_best_model , map_location=torch.device('cpu')))
    logger.info('Loaded PreTrained Model')

    model.eval()
    y_true, y_pred = [], [] 
    if text is None :
        for i , batch in enumerate(test_dataloader):
            sents = batch['text'].to(device)
            entity = batch['label'].to(device)
            _, preds = model(sents,entity)

            targets = entity.cpu().detach().numpy()

            y_true.extend([ent for sen in targets for ent in sen if ent != CFG.entity_pad[1]])
            y_pred.extend([ent for sen in preds for ent in sen])
        pred_id2entity = [idx2entity[id_ent] for id_ent in y_pred]
        true_id2entity = [idx2entity[id_ent] for id_ent in y_true]

        if show_report:
            cls_report = show_report(y_pred=pred_id2entity, y_true=true_id2entity, labels = list(entity2idx.keys()))
            print(cls_report)
        val_f1 = cal_f1score(y_true, y_pred)
        print(val_f1)
    else:
        for i , batch in enumerate(test_dataloader):
            sents = batch['text'].to(device)
            preds = model.inferences(sents) 
            pred_id2entity = [idx2entity[id_ent] for id_ent in preds[0]]
            logger.info(pred_id2entity)

            df =pd.DataFrame()
            df['words'] = np.squeeze(processed_t)
            df['entity'] = pred_id2entity
            df.to_csv(CFG.result_csv,index=False)
            logger.info('Entities and Words are writed into {} folder'.format(CFG.result_csv))

if __name__ =='__main__':
    ap = argparse.ArgumentParser(description='Named Entity Recognition Model')
    ap.add_argument('-i','--input_text', default=None , help='Input senetences for entity recognition')
    args = vars(ap.parse_args())
    main(args['input_text'])


