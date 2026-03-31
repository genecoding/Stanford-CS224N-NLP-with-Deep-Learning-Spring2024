# Implementation of Masked Language Modeling (MLM) for further pretraining.
# Inspired by 
#     https://medium.com/@eraparihar98/understanding-masked-language-modeling-as-a-pretraining-task-4a887ec61c8d


import csv
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from bert_part2 import BertModel
from optimizer import AdamW
from tokenizer import BertTokenizer


TQDM_DISABLE=False


# fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())

def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def load_multitask_data(sentiment_file, paraphrase_file, similarity_file, split='train'):
    """
    Only return sentences for MLM.
    Divide similarity pairs into seperate sentences but keep paraphrase pairs.
    """
    sentiment_data = []
    with open(sentiment_file, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            sent = record['sentence'].lower().strip()
            sentiment_data.append((sent, None))
            
    print(f'Loaded {len(sentiment_data)} {split} examples from {sentiment_file}')
    
    paraphrase_data = []
    with open(paraphrase_file, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            paraphrase_data.append((preprocess_string(record['sentence1']),
                                    preprocess_string(record['sentence2'])))
    
    print(f'Loaded {len(paraphrase_data)} {split} examples from {paraphrase_file}')
    
    similarity_data = []
    with open(similarity_file, 'r') as fp:
        for record in csv.DictReader(fp, delimiter='\t'):
            similarity_data.append((preprocess_string(record['sentence1']), None))
            similarity_data.append((preprocess_string(record['sentence2']), None))
    
    print(f'Loaded {len(similarity_data)} {split} examples from {similarity_file}')
    
    return sentiment_data, paraphrase_data, similarity_data


def mask_tokens(input_ids, tokenizer, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 
    80% MASK, 10% random, 10% original.
    """
    labels = input_ids.clone()
    
    # don't mask special tokens ([CLS], [SEP], [PAD])
    # special_tokens_mask = [
    #     tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) 
    #     for val in input_ids.tolist()
    # ]
    # special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    # tokenizer.get_special_tokens_mask() doesn't work as expected, 
    # so we obtain the mask by taking the union of assigned tokens explicitly
    special_tokens_mask = (
        (input_ids == tokenizer.cls_token_id)
        | (input_ids == tokenizer.sep_token_id)
        | (input_ids == tokenizer.pad_token_id)
    )

    # set probability of special tokens to 0
    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # we only compute loss on masked tokens; CrossEntropyLoss will ignore -100

    # 80% of the time, replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, replace with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # the rest 10% remains original
    return input_ids, labels


class MLMDataset(Dataset):
    def __init__(self, sentiment_data, paraphrase_data, similarity_data, args):
        # datapoints are (sent1, sent2) or (sent, None)
        self.dataset = sentiment_data + paraphrase_data + similarity_data
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        return self.dataset[idx]
        
    def pad_data(self, data):
        batch = []
        for sent1, sent2 in data:
            if sent2:  # concat pair (para)
                encoding = self.tokenizer(sent1, sent2, truncation=True)
            else: # single sent (sst, sts)
                encoding = self.tokenizer(sent1, truncation=True)
            batch.append(encoding)
            
        batch_padded = self.tokenizer.pad(batch, padding=True, return_tensors='pt')
        token_ids = torch.LongTensor(batch_padded['input_ids'])
        attention_mask = torch.LongTensor(batch_padded['attention_mask'])
        
        return token_ids, attention_mask
        
    def collate_fn(self, all_data):
        token_ids, attention_mask = self.pad_data(all_data)
        token_ids, labels = mask_tokens(token_ids, self.tokenizer, self.args.mlm_prob)
        
        batched_data = {
            'token_ids': token_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        return batched_data
    

class MaskedLMBERT(nn.Module):
    '''
    This module use BERT for Masked LM.
    '''
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        self.mlm_head = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                      nn.GELU(),
                                      nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))
        
        # decoder: project to vocabulary size
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)
        # tie the output weights with the input word embeddings, 
        # but use an output-only bias for each token
        self.decoder.weight = self.bert.word_embedding.weight
        nn.init.zeros_(self.decoder.bias)
        
    def forward(self, input_ids, attention_mask):
        hidden_state = self.bert(input_ids, attention_mask)['last_hidden_state']  # (batch, seq_len, hidden_size)
        x = self.mlm_head(hidden_state)
        logits = self.decoder(x)
        return logits
        

def train_mlm(args):
    """
    Train MLM on SST, Quora (for para), STS datasets.
    """
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    
    sst_train_data, para_train_data, sts_train_data = \
        load_multitask_data(args.sst_train, args.para_train, args.sts_train, split='train')
    sst_dev_data, para_dev_data, sts_dev_data = \
        load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='train')

    train_data = MLMDataset(sst_train_data, para_train_data, sts_train_data, args)
    dev_data = MLMDataset(sst_dev_data, para_dev_data, sts_dev_data, args)

    train_dataloader = DataLoader(train_data, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_data.collate_fn)
    dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_data.collate_fn)
    
    # init model
    config = {
        'hidden_size': 768,
        # align with config.py
        'layer_norm_eps': 1e-12,
        'vocab_size': 30522,
    }
    config = SimpleNamespace(**config)
    model = MaskedLMBERT(config)
    model = model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=args.lr)
    best_dev_loss = float('inf')
    patience = 0
    
    # AMP settings
    # A100 supports bfloat16, T4 and V100 don't support bfloat16 
    support_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    amp_dtype = torch.bfloat16 if support_bf16 else torch.float16
    scaler = None if support_bf16 else GradScaler()
    print(f'Support native bfloat16: {support_bf16}, {amp_dtype} is used.')
    
    # run for the specified number of epochs
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            b_ids, b_mask, b_labels = (batch['token_ids'],
                                       batch['attention_mask'], batch['labels'])
            b_ids = b_ids.to(device)
            b_mask = b_mask.to(device)
            b_labels = b_labels.to(device)
            
            with autocast(device_type='cuda', dtype=amp_dtype):
                logits = model(b_ids, b_mask)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), b_labels.view(-1))
            
            optimizer.zero_grad()
            if scaler is None:
                loss.backward()
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
            
        train_loss = train_loss / num_batches
        
        # free memory
        del loss
        torch.cuda.empty_cache()
        
        with autocast(device_type='cuda', dtype=amp_dtype):
            dev_loss = model_eval(dev_dataloader, model, device)
        
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_model(model, optimizer, args, config, args.savepath)
            patience = 0
        else:
            patience += 1

        print(f'Epoch {epoch}: train loss :: {train_loss :.3f}, dev loss :: {dev_loss :.3f}')
        
        if patience == args.patience:
            print('Early Stopping!')
            break

        
@torch.no_grad()
def model_eval(dataloader, model, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE):
        b_ids, b_mask, b_labels = (batch['token_ids'],
                                   batch['attention_mask'], batch['labels'])
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)
        b_labels = b_labels.to(device)
            
        logits = model(b_ids, b_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), b_labels.view(-1))
        
        total_loss += loss.item()
        num_batches += 1
        
    return total_loss / num_batches
    

def get_args():
    parser = argparse.ArgumentParser()
    # sentiment analysis
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    # parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    # paraphrase detection
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    # parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    # semantic textual similarity
    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    # parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    
    parser.add_argument("--mlm_prob", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--savepath", type=str, default="model_further-pretraining.pt")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    print(args)  # show hyperparameters
    print()
    
    seed_everything(args.seed)  # fix the seed for reproducibility
    train_mlm(args)

