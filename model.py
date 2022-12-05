import torch
import torch.nn as nn
from config import CFG
from utils import attention_padding_mask ,attention_padding_mask_infer
from attention import MultiHeadAttention , FeedForward
from crf import CRF



class NamedEntityRecog(nn.Module):
    def __init__(self , num_vocabs,word_embed_dim , lstm_hidden_dim ,num_entities , pretrain_embed ) :
        super(NamedEntityRecog, self).__init__()

        self.num_blocks = CFG.num_blocks
        self.word_pad_idx = CFG.word_pad[1]
        self.ent_pad_idx = CFG.entity_pad[1]
        self.ent_bos_idx = CFG.entity_bos[1]
        self.ent_eos_idx = CFG.entity_eos[1]

        self.embeds = nn.Embedding(num_vocabs , word_embed_dim , padding_idx=0)
        if pretrain_embed is not None:
            self.embeds.weight.data.copy_(torch.from_numpy(pretrain_embed))

        self.lstm = nn.LSTM(word_embed_dim, lstm_hidden_dim , batch_first =True , bidirectional=True)

        for i in range(self.num_blocks):
            self.__setattr__('multihead_attn_{}'.format(i), MultiHeadAttention(model_dim=CFG.model_dim,
                                                                               num_heads=CFG.num_heads,
                                                                               dropout_rate=CFG.dropout_rate,
                                                                               ))
            self.__setattr__('feedforward_{}'.format(i), FeedForward(model_dim=CFG.model_dim,
                                                                     hidden_dim=CFG.ff_hidden_dim,
                                                                     dropout_rate=CFG.dropout_rate))


        self.fc = nn.Linear(CFG.model_dim, num_entities)

        self.crf = CRF(num_entities=num_entities,
                       pad_idx=self.ent_pad_idx,
                       bos_idx=self.ent_bos_idx,
                       eos_idx=self.ent_eos_idx,
                       )

    def forward(self, x, y):
        

        attn_mask = attention_padding_mask(x, y, padding_index=self.word_pad_idx)  # (B, T, T)
        x = self.embeds(x)  # (B, T, D)
        x, _ = self.lstm(x)  # x (B, T, 2 * D/2)

        for i in range(self.num_blocks):
            x, _ = self.__getattr__('multihead_attn_{}'.format(i))(x, x, x, attn_mask=attn_mask)  # (B, T, D)
            x = self.__getattr__('feedforward_{}'.format(i))(x)  # (B, T, D)

        x = self.fc(x)  # x is now emission matrix (B, T, num_entities)
        crf_mask = (y != self.ent_pad_idx).bool()  # (B, T)

        score, path = self.crf.viterbi_decode(x, crf_mask)
        return score, path

    def loss(self, x, y):
        
        attn_mask = attention_padding_mask(x, y, padding_index=self.word_pad_idx)  # (B, T, T)
        
        
        x = self.embeds(x)  # (B, T, D)

        x, _ = self.lstm(x)  # x (B, T, 2 * D/2)

        for i in range(self.num_blocks):
            x, _ = self.__getattr__('multihead_attn_{}'.format(i))(x, x, x, attn_mask=attn_mask)  # (B, T, D)
            x = self.__getattr__('feedforward_{}'.format(i))(x)  # (B, T, D)

        x = self.fc(x)  # x is now emission matrix (B, T, num_entities)

        crf_mask = (y != self.ent_pad_idx).bool()  # (B, T)

        loss = self.crf(x, y, crf_mask)
        return loss

    def inferences(self , x):
        y=x
        attn_mask = attention_padding_mask_infer(x,padding_index=self.word_pad_idx)
        
        x= self.embeds(x)
        x,_ = self.lstm(x)

        for i in range(self.num_blocks):
            x, _ = self.__getattr__('multihead_attn_{}'.format(i))(x, x, x, attn_mask=attn_mask)  # (B, T, D)
            x = self.__getattr__('feedforward_{}'.format(i))(x)  # (B, T, D)

        x = self.fc(x)  # x is now emission matrix (B, T, num_entities)

        crf_mask = (y != self.ent_pad_idx).bool()  # (B, T)

        _, path = self.crf.viterbi_decode(x, crf_mask)
        return path