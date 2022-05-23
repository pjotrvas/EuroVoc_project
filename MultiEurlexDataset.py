from datasets import load_dataset
from torch.utils.data import Dataset
import torch
import pandas as pd
import re
import numpy as np


class MultiEurlexDataset(Dataset):
    def __init__(self, split='train', languages=[], tokenizer=lambda x: x, num_labels=21):
        self.num_labels = num_labels
        dataset = load_dataset('multi_eurlex', 'all_languages')
        dataset_dict = {'celex_id': [],
                        'lang': [],
                        'document': [],
                        'labels': []
                        }

        if isinstance(languages, str):
            languages = [languages]
            print(languages)

        regex1 = re.compile('[0-9\n/()\[\]\':;"\„\“\-»«\’\’\‘\”]')
        regex2 = re.compile('[\ ]+')
        regex3 = re.compile('\ \.')

        for idx, instance in enumerate(dataset[split]):
            step = int(len(dataset[split]) / 10)
            if idx % step == 0:
                percentage = idx / len(dataset[split]) * 100
                print(f'{percentage:.1f}% of dataset loaded')
            for lang in instance['text'].keys():
                if not len(languages) or lang in languages:
                    if instance['text'][lang]:
                        instance['text'][lang] = regex1.sub('', instance['text'][lang])
                        instance['text'][lang] = regex2.sub(' ', instance['text'][lang])
                        instance['text'][lang] = regex3.sub('.', instance['text'][lang])

                        dataset_dict['labels'].append(instance['labels'])
                        dataset_dict['document'].append(instance['text'][lang])
        self.encodings = tokenizer(dataset_dict['document'], padding=True, truncation=True)
        self.labels = dataset_dict['labels']
        self.languages = languages
        print("Loading dataset done")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.multihot_encode(self.labels[idx]))
        return item

    def multihot_encode(self,labels):
        elems = np.zeros(self.num_labels)
        elems[labels] = 1
        return elems

# dataset = MultiEurlexDataset()
# print(dataset.data)
