from MultiEurlexDataset import MultiEurlexDataset
from MultiEvalDataset import MultiEvalDataset
from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizerFast, TrainingArguments, Trainer, DataCollatorWithPadding, TextClassificationPipeline, EarlyStoppingCallback
import transformers
from torch.utils.data import RandomSampler
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
from argparse import ArgumentParser
import torch
import os
import random

import logging

logging.basicConfig(level=logging.DEBUG)
transformers.logging.set_verbosity_debug()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## set seeds
seeds = [7, 13, 42]

languages_dict_train = {
    'germanic': ['en', 'de', 'nl', 'sv', 'da'],
    'romance': ['fr', 'it', 'es', 'ro', 'pt'],
    'slavic': ['pl', 'cs', 'bg', 'sk', 'sl', 'hr'],
    'germanic_': ['en', 'de', 'nl', 'da'],
    'romance_': ['fr', 'it', 'es', 'pt'],
    'slavic_': ['pl', 'cs', 'bg', 'sk', 'sl'],
    'greek': ['el'],
    'uralic': ['hu', 'fi', 'et'],
    'baltic': ['lt', 'lv'],
    'semitic': ['mt'],
}

languages_dict_test = {
    'slavic': ['hr'],
    'germanic': ['sv'],
    'romance': ['ro'],
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base', padding=True, trunaction=True)

collator_fn = DataCollatorWithPadding(tokenizer,return_tensors="pt", padding="max_length", max_length=512)

metric = f1_score


def compute_metrics(eval_pred):
    label_pred, label_true = eval_pred
    sigmoid = nn.Sigmoid()
    label_pred=sigmoid(torch.tensor(label_pred)).numpy()
    label_pred[label_pred < 0.5] = 0
    label_pred[label_pred >= 0.5] = 1
    result={}
    result['f1'] = metric(label_true,label_pred, average='samples')
    return result


for group_target, target in languages_dict_test.items():
    print(target)
    eval_dataset = MultiEurlexDataset(split='validation', languages=target, tokenizer=tokenizer)
    test_dataset = MultiEurlexDataset(split='test', languages=target, tokenizer=tokenizer)
    
    for group, langs in languages_dict_train.items():

        if '_' in group and group_target not in group:
            continue
        
        training_dataset = MultiEurlexDataset(languages=langs, tokenizer=tokenizer)

        for seed in seeds:
            print(f'target: {target[0]}, group: {group}, seed: {seed}')
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
        
            training_args = TrainingArguments(
                output_dir=os.path.join('./results', target[0], group, f'seed_{seed}'), # output directory
                num_train_epochs=5,              # total number of training epochs
                per_device_train_batch_size=8,  # batch size per device during training
                per_device_eval_batch_size=16,   # batch size for evaluation
                warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
                weight_decay=0.001,               # strength of weight decay
                logging_dir='./logs',            # directory for storing logs
                logging_strategy='steps',
                logging_steps=100,
                learning_rate=1e-5,
                save_strategy='epoch',
                evaluation_strategy='epoch',
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                lr_scheduler_type='polynomial',
                dataloader_num_workers=24
            )

            model = XLMRobertaForSequenceClassification.from_pretrained('xlm-roberta-base', num_labels=21, problem_type="multi_label_classification", )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=training_dataset,
                eval_dataset=eval_dataset,
                
                compute_metrics=compute_metrics,
                data_collator=collator_fn,
                callbacks=[
                    EarlyStoppingCallback(early_stopping_patience=3),
                ]
            )

            trainer.train()
            
            path = os.path.join('logs2', f'{target[0]}_{group}_{seed}.out')

            with open(path, 'w') as f:
                f.write(trainer.predict(test_dataset))