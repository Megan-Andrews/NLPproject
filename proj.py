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


# class RedditModel():

class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long),
            'tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].flatten())  # For visualization
        }




def read_data():
    # Load pre-trained DistilBERT model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', output_attentions=True)
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2, output_attentions=True)

    # Define optimizer and learning rate
    optimizer = AdamW(model.parameters(), lr=opts.lr)

    # Define training parameters
    epochs = opts.epochs
    batch_size = opts.batchsize

    max_len = 128  # Maximum sequence length for DistilBERT

    # Move model to appropriate device (GPU if available, else CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    df = pd.read_csv(opts.inputfile, names=['clean_text', 'is_depression'], header=0)
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['clean_text'], df['is_depression'], test_size=0.2, random_state=42)
    # print(train_texts)
    train_dataset = RedditDataset(train_texts.reset_index(drop=True), train_labels.reset_index(drop=True), tokenizer, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = RedditDataset(val_texts.reset_index(drop=True), val_labels.reset_index(drop=True), tokenizer, max_len)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # print(df)

    # Iterate over epochs
    # Iterate over epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        predictions = []
        true_labels = []

        # Wrap train_loader with tqdm
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', unit='batch')):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            attention_weights = outputs.attentions  # This assumes your model outputs attention weights
            # print(outputs,attention_weights)
            batch_predictions = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(batch_predictions)
            true_labels.extend(labels.cpu().numpy())
            
            # print(f'Accuracy: {np.mean(batch_predictions==labels.cpu().numpy())}')
            # Print some sample outputs with predicted class and correct class
            # num_samples = 5
            # print(f'Sample outputs (predicted class, correct class):')
            # for i in range(num_samples):
            #     print(f'Predicted: {batch_predictions[i]}, Correct: {labels[i]}, Truth: {batch_predictions[i]==labels[i]}')


            layer_index = 0  # Choose which layer's attention weights to visualize
            attention_weights_layer = attention_weights[0][layer_index]

            # Assuming you have a list of tokens corresponding to the input text
            tokens = batch['tokens']

            # Visualize attention weights for a single example in the batch
            example_index = 0  # Choose which example's attention weights to visualize
            #print(attention_weights_layer.shape)
            plt.figure(figsize=(10, 10))
            plt.imshow(attention_weights_layer[example_index].cpu().detach().numpy(), cmap='hot', interpolation='nearest')
            plt.xticks(range(len(tokens)), tokens, rotation=45)
            plt.yticks(range(len(tokens)), tokens)
            plt.xlabel('Output Tokens')
            plt.ylabel('Input Tokens')
            plt.title(f'Attention Weights (Layer {layer_index})')
            plt.colorbar()
            plt.show()

            loss.backward()
            optimizer.step()
            # print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}')
            # Print sample outputs after the 50th batch
            if batch_idx >= 2:
                break

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
    