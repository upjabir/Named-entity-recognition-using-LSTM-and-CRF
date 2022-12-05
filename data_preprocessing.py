import pandas as pd
from config import CFG
from sklearn.model_selection import train_test_split
import pickle
from utils import load_dataset

def grp_split_csv(file,training_data=False):
    df = pd.read_csv(file)
    #df = df.sample(frac=1)
    agg_func = lambda s: [(w,p) for w,p, in zip(s["word"].values.tolist(),
                                                        s["tag"].values.tolist())]
    # grp_data=df.groupby(['senetence']).reset_index()
    # grp_data.to_csv('grouped.csv',index=False)

    agg_data=df.groupby(['senetence']).apply(agg_func).reset_index().rename(columns={0:'Sentence_POS_Tag_Pair'})
    agg_data['Sentence']=agg_data['Sentence_POS_Tag_Pair'].apply(lambda sentence:" ".join([s[0] for s in sentence]))
    agg_data['Tag']=agg_data['Sentence_POS_Tag_Pair'].apply(lambda sentence:" ".join([s[1] for s in sentence]))
    agg_data['tag_list']=agg_data['Tag'].apply(lambda x:x.split())
    agg_data['tokenised_sentences']=agg_data['Sentence'].apply(lambda x:x.split())
    agg_data = agg_data[['tokenised_sentences','tag_list']]
    train_data = agg_data[['tokenised_sentences', 'tag_list']].apply(tuple, axis=1).tolist()
    train, test =train_test_split(agg_data, test_size=0.3, random_state=42)
    val , test = train_test_split(test, test_size=0.3, random_state=42)
    train.to_csv(CFG.train_data,index=False)
    test.to_csv(CFG.test_data,index=False)
    val.to_csv(CFG.val_data,index=False)

    if training_data:
        return train_data
    else:
        return 1

def build_lookup(data):

    word_to_ix = {sign: idx for sign, idx in [CFG.word_pad, CFG.word_oov]}
    tag_to_ix = {sign: idx for sign, idx in [CFG.entity_pad, CFG.entity_bos, CFG.entity_eos]}

    for sentence, tags in data:
        
        for word, tag in zip(sentence, tags):
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    return word_to_ix, tag_to_ix 





if __name__ =='__main__':
    whole_data = grp_split_csv(CFG.raw_data_dir,training_data=True)  
    #whole_data = load_dataset('data/new_train.csv')
    #print(whole_data[0])
    word2idx , entity2idx = build_lookup(whole_data)

    lookup = dict()
    lookup['word2idx'] = word2idx
    lookup['entity2idx'] = entity2idx

    with open(CFG.lookup_path, 'wb') as fout:
            pickle.dump(lookup, fout)


