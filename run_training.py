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

import logging

logging.basicConfig(level=logging.DEBUG)
transformers.logging.set_verbosity_debug()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

germanic = ['en', 'de', 'nl', 'sv', 'da']
romance = ['fr', 'it', 'es', 'ro', 'pt']
slavic = ['pl', 'cs', 'bg', 'sk', 'sl']
greek = ['el']
uralic = ['hu', 'fi', 'et']
baltic = ['lt', 'lv']
semitic = ['mt']
croatian = ['hr']

device='cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base', padding=True, trunaction=True)
training_dataset = MultiEurlexDataset(languages = croatian, tokenizer=tokenizer)
eval_dataset = MultiEurlexDataset(split='validation',languages = ['hr'], tokenizer=tokenizer)
test_dataset = MultiEurlexDataset(split='test',languages = ['hr'], tokenizer=tokenizer)

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

training_args = TrainingArguments(
    output_dir='./results/croatian_10',          # output directory
    num_train_epochs=10,              # total number of training epochs
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
print(trainer.predict(test_dataset))