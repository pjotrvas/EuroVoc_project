from datasets import load_dataset
from torch.utils.data import Dataset
import pandas as pd


class MultiEurlexDataset(Dataset):
    def __init__(self,split='train',language='all_languages'):
        dataset = load_dataset('multi_eurlex', language)
        dataset_dict = {'celex_id': [],
                'lang': [],
                'document': [],
                'labels': []
                }

        for idx, instance in enumerate(dataset[split]):
            step = int(len(dataset[split])  / 10)
            if idx % step == 0:
                percentage = idx / len(dataset[split]) * 100
                print(f'{percentage:.4f}% of dataset loaded')
            for lang in instance['text'].keys():
                if instance['text'][lang]:
                    dataset_dict['celex_id'].append(instance['celex_id'])
                    dataset_dict['lang'].append(lang)
                    dataset_dict['labels'].append(instance['labels'])
                    dataset_dict['document'].append(instance['text'][lang])
                
        flat_dataset =  pd.DataFrame.from_dict(dataset_dict)
        self.data = flat_dataset
        self.language = language

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data.loc[idx, 'lang'], self.data.loc(idx, 'labels')

#dataset = MultiEurlexDataset()
#print(dataset.data)