import torch
from torch.utils.data import Dataset
from config import CFG

class NERDataset(Dataset):
    def __init__(self , word_vocab , label_vocab , training_data,inference=False) :
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.training_data = training_data
        self.inference = inference

    def __len__(self):
        return len(self.training_data)
    
    def __getitem__(self, index) :

        if not self.inference:
            text_id =[]
            label_id =[]

            text = self.training_data[index][0]
            label = self.training_data[index][1]
            
            #print(text ,label)
        
            text = text[:CFG.max_len]
            label = label[:CFG.max_len]

            for word in text:
                text_id.append(self.word_vocab[word])
            for lab in label:
                label_id.append(self.label_vocab[lab])
            

            #mask = [1] * len(text)
            padding_length = CFG.max_len - len(text)
            if padding_length > 0:
                text_id=text_id + ([0]*padding_length)
                label_id=label_id + ([0]*padding_length)
                #mask = mask + ([0]*padding_length)
            
            return {
                'text': torch.tensor(text_id, dtype=torch.long),
                'label': torch.tensor(label_id, dtype=torch.long),
                #'tag_mask': torch.tensor(mask, dtype=torch.long),
                
            }
        else:
            text_id =[]
            text = self.training_data[index]
            #print(text)
            text = text[:CFG.max_len]

            for word in text:
                text_id.append(self.word_vocab[word])

            padding_length = CFG.max_len - len(text)
            if padding_length > 0:
                text_id=text_id + ([0]*padding_length)
            
            return {'text': torch.tensor(text_id, dtype=torch.long)}




        
