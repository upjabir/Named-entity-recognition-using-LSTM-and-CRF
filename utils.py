import numpy as np
import torch
from sklearn.metrics import f1_score , classification_report
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from ast import literal_eval

def load_pretrain_emb(embedding_path):
    embedd_dict = dict()
    with open(embedding_path, 'r', encoding="utf8") as file:
        for line in file:
            records = line.split()
            word = records[0]
            vector_dimensions = np.asarray(records[1:], dtype='float32')
            embedd_dict [word] = vector_dimensions
    return embedd_dict



def build_pretrain_embedding(embedding_path, word_vocab, embedd_dim):
    embedd_dict = dict()
    if embedding_path is not None:
        embedd_dict = load_pretrain_emb(embedding_path)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([len(word_vocab), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for word, index in word_vocab.items():
        if word in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word]
            perfect_match += 1
        elif word.lower() in embedd_dict:
            pretrain_emb[index, :] = embedd_dict[word.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1

    pretrain_emb[0, :] = np.zeros((1, embedd_dim))
    pretrained_size = len(embedd_dict)
    print("Embedding:\n     pretrain word:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
        pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / len(word_vocab)))
    return pretrain_emb


def load_dataset(data):
    df = pd.read_csv(data)
    df['tokenised_sentences'] = df['tokenised_sentences'].apply(literal_eval)
    df['tag_list'] = df['tag_list'].apply(literal_eval)
    train_data = df[['tokenised_sentences', 'tag_list']].apply(tuple, axis=1).tolist()
    return train_data


def attention_padding_mask(q, k, padding_index=0):
    
    mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    return mask


def attention_padding_mask_infer(q, padding_index=0):
    mask = q.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    return mask




def cal_accuracy(predicts, targets, ignore_index=None):
   
    assert predicts.shape == targets.shape, 'predicts and targets should have same shape'

    if ignore_index is not None:
        valid = targets != ignore_index
        predicts = predicts[valid]  # would be flattened and with valid positions chosen (*)
        targets = targets[valid]  # would be flattened and with valid positions chosen (*)

    return np.sum(predicts == targets) / predicts.size


def cal_f1score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

def show_report(y_true , y_pred , labels):
    return classification_report(y_pred, y_true, labels)


class EarlyStopping:
    def __init__(self, monitor='loss', min_delta=0., patience=0):
        """EarlyStopping
        Args:
            monitor (str): quantity to be monitored. 'loss' or 'acc'
            min_delta (float): minimum change in the monitored quantity to qualify as an improvement
                i.e. an absolute change of less than min_delta, will count as no improvement.
            patience (int): number of epochs with no improvement after which training will be stopped
        """

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self._wait = 0
        self._best = None
        self._best_epoch = None

        if 'loss' in self.monitor:
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1

    def judge(self, epoch, value):
        current = value

        if self._best is None:
            self._best = current
            self._best_epoch = epoch
            return

        if self.monitor_op(current - self.min_delta, self._best):
            self._best = current
            self._best_epoch = epoch
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                return True

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_val(self):
        return self._best
