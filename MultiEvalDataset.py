from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd
import re


class MultiEurlexDataset(Dataset):
    def __init__(self, split='train', languages=[]):
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
                print(f'{percentage:.4f}% of dataset loaded')
            for lang in instance['text'].keys():
                if not len(languages) or lang in languages:
                    if instance['text'][lang]:
                        instance['text'][lang] = regex1.sub('', instance['text'][lang])
                        instance['text'][lang] = regex2.sub(' ', instance['text'][lang])
                        instance['text'][lang] = regex3.sub('.', instance['text'][lang])

                        dataset_dict['celex_id'].append(instance['celex_id'])
                        dataset_dict['lang'].append(lang)
                        dataset_dict['labels'].append(instance['labels'])
                        dataset_dict['document'].append(instance['text'][lang])

        flat_dataset = pd.DataFrame.from_dict(dataset_dict)
        self.data = flat_dataset
        self.languages = languages

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, 'document'], self.data.loc[idx, 'labels']

# dataset = MultiEurlexDataset()
# print(dataset.data)