import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle

os.environ['TOKENIZERS_PARALLELISM'] = '0'


# Zero-Shot Stance Detection: A Dataset and Model using Generalized Topic Representations
class VASTZeroFewShot(Dataset):
    def __init__(self, phase, target_label_list):
        path = 'data/vast/'
        if phase in ['train', 'test']:
            file_path = f'{path}/vast_{phase}.csv'
        else:
            file_path = f'{path}/vast_dev.csv'
        df = pd.read_csv(file_path)
        print(f'# of {phase} examples: {df.shape[0]}')

        topics = df['topic_str'].tolist()
        tweets = df['text_s'].tolist()
        stances = df['label'].tolist()
        new_topic = df["new_topic"].tolist()
        if phase == 'test':
            few_shot = df['seen?'].tolist()
            qte = df['Qte'].tolist()
            sarc = df['Sarc'].tolist()
            imp = df['Imp'].tolist()
            mls = df['mlS'].tolist()
            mlt = df['mlT'].tolist()
        else:
            few_shot = np.zeros(df.shape[0])
            qte = np.zeros(df.shape[0])
            sarc = np.zeros(df.shape[0])
            imp = np.zeros(df.shape[0])
            mls = np.zeros(df.shape[0])
            mlt = np.zeros(df.shape[0])

        # os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from transformers import AutoTokenizer

        tokenizer_shared = AutoTokenizer.from_pretrained('bert-base-uncased')

        tokenizer_wiki = AutoTokenizer.from_pretrained('bert-base-uncased')

        tokenizer_google = AutoTokenizer.from_pretrained("bert-base-uncased")

        tokenizer_tweet = AutoTokenizer.from_pretrained('vinai/bertweet-base')

        wiki_dict = pickle.load(open(f'{path}/wiki_dict.pkl', 'rb'))
        wiki_summaries = df['new_topic'].map(wiki_dict).tolist()

        google_dict = pickle.load(open(f'{path}/google_dict.pkl', 'rb'))
        google_summaries = df['new_topic'].map(google_dict).tolist()

        tweets_targets = [f'text: {x} target: {y}' for x, y in zip(tweets, topics)]

        summaries = [f'g: {x} w: {y}' for x, y in zip(google_summaries, wiki_summaries)]

        encodings_shared = tokenizer_shared(tweets_targets, summaries, padding=True, truncation=True)
        encodings_wiki = tokenizer_wiki(wiki_summaries, padding=True, truncation=True)
        encodings_google = tokenizer_google(google_summaries, padding=True, truncation=True)
        encodings_tweet = tokenizer_tweet(tweets, topics, padding=True, truncation=True)

        # encodings for the texts and tweets
        input_ids_shared = torch.tensor(encodings_shared['input_ids'], dtype=torch.long)
        attention_mask_shared = torch.tensor(encodings_shared['attention_mask'], dtype=torch.long)
        token_type_ids_shared = torch.tensor(encodings_shared['token_type_ids'], dtype=torch.long)

        # encodings for wiki summaries
        input_ids_wiki = torch.tensor(encodings_wiki['input_ids'], dtype=torch.long)
        attention_mask_wiki = torch.tensor(encodings_wiki['attention_mask'], dtype=torch.long)

        # encodings for google summaries
        input_ids_google = torch.tensor(encodings_google['input_ids'], dtype=torch.long)
        attention_mask_google = torch.tensor(encodings_google['attention_mask'], dtype=torch.long)

        input_ids_tweet = torch.tensor(encodings_tweet['input_ids'], dtype=torch.long)
        attention_mask_tweet = torch.tensor(encodings_tweet['attention_mask'], dtype=torch.long)
        token_type_ids_tweet = torch.tensor(encodings_tweet['token_type_ids'], dtype=torch.long)

        stances = torch.tensor(stances, dtype=torch.long)
        target_label2id = {data: i for i, data in enumerate(target_label_list)}
        # 只需要把new_topic放进来得到一个序列
        target_labels_id = [target_label2id[item] for item in new_topic]
        print(
            f'max len: {input_ids_shared.shape[1]}, max len wiki: {input_ids_wiki.shape[1]}, max len tweet: {input_ids_tweet.shape[1]}')

        self.phase = phase
        self.input_ids_shared = input_ids_shared
        self.attention_mask_shared = attention_mask_shared
        self.token_type_ids_shared = token_type_ids_shared
        self.mlt = mlt
        self.input_ids_wiki = input_ids_wiki
        self.attention_mask_wiki = attention_mask_wiki
        self.input_ids_google = input_ids_google
        self.attention_mask_google = attention_mask_google
        self.input_ids_tweet = input_ids_tweet
        self.attention_mask_tweet = attention_mask_tweet
        self.token_type_ids_tweet = token_type_ids_tweet
        self.stances = stances
        self.target_labels_id = target_labels_id
        self.few_shot = few_shot
        self.qte = qte
        self.sarc = sarc
        self.imp = imp
        self.mls = mls

    def __getitem__(self, index):
        item = {
            'input_ids_shared': self.input_ids_shared[index],
            'attention_mask_shared': self.attention_mask_shared[index],
            'token_type_ids_shared': self.token_type_ids_shared[index],
            'input_ids_wiki': self.input_ids_wiki[index],
            'attention_mask_wiki': self.attention_mask_wiki[index],
            'input_ids_google': self.input_ids_google[index],
            'attention_mask_google': self.attention_mask_google[index],
            'input_ids_tweet': self.input_ids_tweet[index],
            'attention_mask_tweet': self.attention_mask_tweet[index],
            'token_type_ids_tweet': self.token_type_ids_tweet[index],
            'stances': self.stances[index],
            'target_labels_id': self.target_labels_id[index],
            'few_shot': self.few_shot[index],
            'qte': self.qte[index],
            'sarc': self.sarc[index],
            'imp': self.imp[index],
            'mls': self.mls[index],
            'mlt': self.mlt[index],
        }
        return item

    def __len__(self):
        return self.stances.shape[0]


def data_loader(data, phase, batch_size, target_label_list, n_workers=4):
    shuffle = True if phase == 'train' else False
    dataset = VASTZeroFewShot(phase, target_label_list)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=n_workers)
    return loader
