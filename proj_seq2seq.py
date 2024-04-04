from datasets import load_dataset
from transformers import BartTokenizer, BartForConditionalGeneration
import argparse, os, string, sys
import torch
from pathlib import Path
import random 

import argparse, os, string, sys
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import json

import gc
torch.cuda.empty_cache()
gc.collect()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvoDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def create_text(self, idx):
        """
        Extracts a segment of the chat up to a randomly selected system message 
        occurring after the third message in the conversation, along with the target system message.
        
        Parameters:
        - idx: Index of the conversation in the dataset.
        
        Returns:
        - processed_chat: A string containing the conversation up to the target system message.
        - target_message: The selected target system message.
        """
        
        chat = []
        sys_idx = []

        # Attempt to load the JSON data safely
        try:
            dialog = json.loads(self.texts[idx]['text'])['dialog']
        except (KeyError, ValueError) as e:
            print(f"Error loading conversation data: {e}")
            return "", ""


        for e, i in enumerate(dialog):
            if i['speaker'] == 'usr':
                chat.append(i['text'])
            if i['speaker'] == 'sys':
                sys_idx.append(e)
                chat.append(f"[{i['strategy']}] " + i['text'])
            continue
    
        target_idx = int(random.choice(sys_idx))
        while target_idx < 4:
            target_idx = int(random.choice(sys_idx))

        #target_idx = sys_idx[3]
        processed_chat = "\n".join([c for e, c in enumerate(chat[:target_idx]) if e not in sys_idx])

        # print(processed_chat + "\n\n")
        #print(chat[target_idx])

        return  "Provide suggestions, affirmations or reflection of feelings for the following person's needs. "+processed_chat, chat[target_idx]

    def __getitem__(self, idx):
        processed_chat, target_message = self.create_text(idx)

        # Tokenize the processed chat
        inputs = self.tokenizer(
            processed_chat,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )

        # Tokenize the target message
        targets = self.tokenizer(
            target_message,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors='pt',
        )

        # Since the model expects labels to calculate loss, create labels by shifting the targets to the right
        # This will be automatically handled if using a model that supports labels (like T5 or BART from Hugging Face)
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': targets['input_ids'].flatten(),
        }

        return inputs




def read_data():
    # Load pre-trained DistilBERT model and tokenizer
    # tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', output_attentions=True)
    # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, output_attentions=True)
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

    # Define optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=opts.lr)

    # Define training parameters
    epochs = opts.epochs
    batch_size = opts.batchsize

    max_len = 1024  # Maximum sequence length for DistilBERT

    # Move model to appropriate device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    dataset = load_dataset("thu-coai/esconv")

    train_dataset = ConvoDataset(dataset["train"], tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = ConvoDataset(dataset["validation"], tokenizer, max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Iterate over epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Wrap train_loader with tqdm
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            # output_token_ids = model.generate(input_ids)
            # decoded_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_token_ids]
            # for text in decoded_texts:
            #     print(text)
            # decoded_labels = [tokenizer.decode(ids, skip_special_tokens=True) for ids in labels]
            # for label in decoded_labels:
            #     print(label)

            loss = outputs.loss
            
            
            total_loss += loss.item()

            
        avg_train_loss = total_loss / len(train_loader)

        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}')








if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputfile", dest="inputfile",
                            default=os.path.join('data', 'input', 'ddr.csv'),
                             help="produce table to text output for these input tables")
    argparser.add_argument("-m", "--modelfile", dest="modelfile",
                            default=os.path.join('data', 'peft'),
                            help="filename without suffix for model files")
    argparser.add_argument("-s", "--modelsuffix", dest="modelsuffix", default='.pt',
                            help="filename suffix for model files")
    argparser.add_argument("-M", "--basemodel", dest="basemodel",
                            default='distilgpt2',
                            help="The base huggingface pretrained model to be used as the encoder.")
    argparser.add_argument("-e", "--epochs", dest="epochs", type=int, default=1,
                            help="number of epochs [default: 1]")
    argparser.add_argument("-b", "--batchsize", dest="batchsize", type=int, default=16,
                            help="batch size [default: 16]")
    argparser.add_argument("-r", "--lr", dest="lr", type=float, default=5e-5,
                            help="the learning rate used to finetune the BERT-like encoder module.")
    argparser.add_argument("-f", "--force", dest="force", action="store_true", default=False,
                            help="force training phase (warning: can be slow)")
    argparser.add_argument("-l", "--logfile", dest="logfile", default=None,
                            help="log file for debugging")
    opts = argparser.parse_args()
    
    read_data()
    

################## Developing Summarization ###################
"""
from transformers import BartTokenizer, BartForConditionalGeneration
dataset = load_dataset("thu-coai/esconv")
# Load BART
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
model.to(device)
print("loaded model")

chat = []
sys_idx = []
for e, i in enumerate(json.loads(dataset['test'][0]['text'])['dialog']):
    if i['speaker'] == 'usr':
        #chat.append('[Target] ' + i['text'])  
        chat.append(i['text'])
    if i['speaker'] == 'sys':
        #if e > 3:
        #    sys_idx.append(e)
        #chat.append(f" [Context] [{i['strategy']}] " + i['text'])
       continue
    
# target_idx = int(random.choice(sys_idx))
# processed_chat = "\n".join(chat[:target_idx])

# print(processed_chat + "\n\n")
#print(chat[target_idx])
processed_chat = "\n".join(chat)
# Tokenize and generate summary
inputs = tokenizer(processed_chat, return_tensors="pt", max_length=1024, truncation=True)
inputs.to(device)
summary_ids = model.generate(inputs['input_ids'], max_length=200, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(summary)"""