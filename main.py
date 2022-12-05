import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
from dataset import NERDataset 
import pickle
from config import CFG
import time
from model import NamedEntityRecog
from torch.utils.tensorboard import SummaryWriter
import logging
import torch.optim as optim
from utils import EarlyStopping ,load_dataset , build_pretrain_embedding
from train import train_and_eval 
import pytorch_warmup as warmup

logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

data_path = Path('data')
train_file = data_path/'train.csv'

with open(CFG.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)

word2idx = lookup['word2idx']
entity2idx = lookup['entity2idx']

num_vocabs = len(word2idx)
num_entities = len(entity2idx)

print(num_vocabs , num_entities)

training_data = load_dataset(CFG.train_data)
eval_data = load_dataset(CFG.test_data)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info("Training in {}".format(device))


train_dataset = NERDataset(word2idx, entity2idx, training_data)
evaluation_dataset = NERDataset(word2idx,entity2idx,eval_data)

emb_begin = time.time()
pretrain_word_embedding = build_pretrain_embedding(CFG.pretrained_embed_path, word2idx, CFG.word_embed_dim)
emb_end = time.time()
emb_min = (emb_end - emb_begin) % 3600 // 60
logger.info('build pretrain embed cost {}m with shape {}'.format(emb_min,pretrain_word_embedding.shape))

train_dataloader = DataLoader(train_dataset,batch_size=CFG.batch_size)
eval_dataloader = DataLoader(evaluation_dataset,batch_size=CFG.batch_size)

model = NamedEntityRecog(num_vocabs , CFG.word_embed_dim , CFG.lstm_hidden_dim,num_entities , pretrain_embed=pretrain_word_embedding).to(device)

logger.info('Model Start training')
writer = SummaryWriter()
batch_num = 0
earlystop = EarlyStopping(monitor='acc', min_delta=CFG.min_delta, patience=CFG.patience)
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=0.01)
num_steps = len(train_dataloader) * CFG.num_epochs
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps)
# lr_decay = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=CFG.lr_decay_factor, verbose=True,
#                                                     patience=0, min_lr=CFG.min_lr)
warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)


for epoch in range(1, CFG.num_epochs + 1):
        epoch_begin = time.time()
        val_f1 = train_and_eval(train_dataloader,eval_dataloader,model,optimizer,batch_num,writer,device ,epoch , lr_scheduler, warmup_scheduler)
        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        logger.info('Train and Eval{}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time / 60), int(cost_time % 60)))
        if earlystop.judge(epoch, val_f1):
                logger.info('Early stop at epoch {}, with val F1-Score {}'.format(epoch, val_f1))
                logger.info('Best perform epoch: {}, with best F1-Score {}'.format(earlystop.best_epoch, earlystop.best_val))
                break