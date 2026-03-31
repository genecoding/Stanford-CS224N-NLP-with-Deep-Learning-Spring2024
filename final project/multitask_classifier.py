'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# from bert import BertModel
from bert_part2 import BertModel
from optimizer import AdamW
from tqdm import tqdm
from dora import apply_dora_to_all, merge_and_unload_all
from loss import CoSENTLoss, AnglELoss
from grad import PCGrad, CAGrad
from smart import SMARTLoss, BPPOptimization
from sampler import AnnealedSampler, SquareRootSampler


from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    SentencePairCatDataset,
    SentencePairCatTestDataset,
    load_multitask_data
)

from evaluation import (
    model_eval_sst, model_eval_para, model_eval_sts, 
    model_eval_multitask, model_eval_test_multitask
)


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


TQDM_DISABLE=False
BERT_HIDDEN_SIZE = 768
N_SENTIMENT_CLASSES = 5

support_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
amp_dtype = torch.bfloat16 if support_bf16 else torch.float16


class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.
        ### TODO
        # sharing weights between BERT and task heads
        self.shared_interm = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                           nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                                           nn.GELU(),
                                           nn.Dropout(config.hidden_dropout_prob))
                                                         
        # for sentiment analysis
        self.sentiment_head = nn.Sequential(nn.Dropout(config.hidden_dropout_prob),
                                            nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES))

        # for paraphrase detection
        self.paraphrase_head = nn.Sequential(nn.Dropout(config.hidden_dropout_prob),
                                             nn.Linear(config.hidden_size, 1))
        
        self.sts_criterion = CoSENTLoss()


    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).
        ### TODO
        return self.bert(input_ids, attention_mask)['pooler_output']
        

    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''
        ### TODO
        sent_embeds = self.forward(input_ids, attention_mask)
        sent_embeds = self.shared_interm(sent_embeds)
        logits = self.sentiment_head(sent_embeds)  # (batch, 5)
        return logits
        

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                        #   input_ids_2, attention_mask_2
                           ):
        '''Given a batch of concat of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''
        ### TODO
        sent_embeds = self.forward(input_ids_1, attention_mask_1)
        sent_embeds = self.shared_interm(sent_embeds)
        logits = self.paraphrase_head(sent_embeds)  # (batch, 1)
        return logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''
        ### TODO
        sent_embeds1 = self.forward(input_ids_1, attention_mask_1)
        sent_embeds2 = self.forward(input_ids_2, attention_mask_2)
        similarity = self.sts_criterion.get_similarity(sent_embeds1, sent_embeds2)
        return similarity #* 5.0  # don't need to scale; scaling doesn't affect correlation
    
    
    def forward_with_embeds(self, embeds, attention_mask, task):
        """Pass embeddings directly to BERT"""
        sent_embeds = self.bert(embeds, attention_mask, with_embeds=True)['pooler_output']
        
        if task == 'sst':
            sent_embeds = self.shared_interm(sent_embeds)
            return self.sentiment_head(sent_embeds)
        elif task == 'para':
            sent_embeds = self.shared_interm(sent_embeds)
            return self.paraphrase_head(sent_embeds)
        elif task == 'sts':
            return sent_embeds




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
    
    
def load_further_pretrain_model(mlm_filepath, multitask_config):
    saved = torch.load(mlm_filepath, weights_only=False)
    
    # use config of MultitaskBERT to initialize MultitaskBERT
    model = MultitaskBERT(multitask_config)
    
    # load BERT part of MLM
    mlm_state_dict = saved['model']
    bert_only_weights = {k: v for k, v 
                         in mlm_state_dict.items() if k.startswith('bert.')}
    
    # strict=False, to ignore the other weights of MultitaskBERT
    model.load_state_dict(bert_only_weights, strict=False)
    
    print(f'Load further pretrained BERT weights from {mlm_filepath}')
    return model


def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
    '''
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')

    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels, para_train_data, sts_train_data = \
        load_multitask_data(args.sst_train, args.para_train, args.sts_train, split ='train')
    sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
        load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sst_dev_data.collate_fn)
    
    para_train_data = SentencePairCatDataset(para_train_data, args, isRegression=True)
    para_dev_data = SentencePairCatDataset(para_dev_data, args, isRegression=True)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=args.batch_size,
                                       collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                     collate_fn=para_dev_data.collate_fn)
    
    sts_train_data = SentencePairDataset(sts_train_data, args, isRegression=True)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=args.batch_size,
                                      collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                    collate_fn=sts_dev_data.collate_fn)

    loaders = {
        'sst': sst_train_dataloader, 
        'para': para_train_dataloader, 
        'sts': sts_train_dataloader
    }
    task_keys = ['sst', 'para', 'sts']
    task_iters = {k: iter(loaders[k]) for k in task_keys}
    loader_sizes = [len(loaders[k]) for k in task_keys]
    sst_iter = iter(sst_train_dataloader)
    para_iter = iter(para_train_dataloader)
    sts_iter = iter(sts_train_dataloader)
    
    # define epoch length by the middle dataset len
    total_steps = len(sst_train_dataloader)
                                    
    # Init model.
    config = {
        'hidden_dropout_prob': args.hidden_dropout_prob,
        'num_labels': num_labels,
        'hidden_size': 768,
        'data_dir': '.',
        'fine_tune_mode': args.fine_tune_mode,
        'layer_norm_eps': 1e-12,
    }

    config = SimpleNamespace(**config)

    if args.loadpath:
        # load further pretrained model
        model = load_further_pretrain_model(args.loadpath, config)
    else:
        model = MultitaskBERT(config)
    
    if args.use_dora:
        apply_dora_to_all(model.bert, rank=args.rank, alpha=args.alpha)

    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0
    patience = 0
    
    # AMP settings
    # A100 supports bfloat16, T4 and V100 don't support bfloat16 
    # support_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    # amp_dtype = torch.bfloat16 if support_bf16 else torch.float16
    scaler = None if support_bf16 else GradScaler()
    print(f'Support native bfloat16: {support_bf16}, {amp_dtype} is used.')
    
    cagrad = CAGrad(optimizer, scaler, args.cagrad_c)
    smart_reg = SMARTLoss(model)
    # bpp_opt = BPPOptimization(model)
    sampler = AnnealedSampler(loader_sizes, args.epochs, args.max_decay_rate)

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        probs = sampler.get_probabilities(epoch)
        
        for _ in tqdm(range(total_steps), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            for task in task_keys:
                try:
                    batch = next(task_iters[task])
                except StopIteration:
                    task_iters[task] = iter(loaders[task])
                    batch = next(task_iters[task])
                
                if task == 'sst':
                    b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
                    b_ids = b_ids.to(device)  # (batch, seq_len)
                    b_mask = b_mask.to(device)  # (batch, seq_len)
                    b_labels = b_labels.to(device)  # (batch)
            
                    with autocast(device_type='cuda', dtype=amp_dtype):
                        logits = model.predict_sentiment(b_ids, b_mask)
                        loss_sst = F.cross_entropy(logits, b_labels)
                        loss_sst += smart_reg(b_ids, b_mask, 'sst', logits.detach())
                        # loss_sst += bpp_opt(b_ids, b_mask, 'sst', logits)
                        
                elif task == 'para':
                    b_ids, b_mask, b_labels = (batch['token_ids'], batch['attention_mask'], batch['labels'])
                    b_ids = b_ids.to(device)
                    b_mask = b_mask.to(device)
                    b_labels = b_labels.to(device)
                    
                    with autocast(device_type='cuda', dtype=amp_dtype):
                        logits = model.predict_paraphrase(b_ids, b_mask)
                        loss_para = F.binary_cross_entropy_with_logits(logits.view(-1), b_labels.float())
                        loss_para += smart_reg(b_ids, b_mask, 'para', logits.detach())
                        # loss_para += bpp_opt(b_ids, b_mask, 'para', logits)
                        
                elif task == 'sts':
                    (b_ids1, b_mask1, 
                     b_ids2, b_mask2, b_labels) = (batch['token_ids_1'], batch['attention_mask_1'], 
                                                   batch['token_ids_2'], batch['attention_mask_2'], batch['labels'])
                    b_ids1 = b_ids1.to(device)
                    b_ids2 = b_ids2.to(device)
                    b_mask1 = b_mask1.to(device)
                    b_mask2 = b_mask2.to(device)
                    b_labels = b_labels.to(device) #/ 5.0  # normalize is not necessary
                    
                    with autocast(device_type='cuda', dtype=amp_dtype):
                        sent_embeds1 = model(b_ids1, b_mask1)
                        sent_embeds2 = model(b_ids2, b_mask2)
                        loss_sts = model.sts_criterion(sent_embeds1, sent_embeds2, b_labels)
                        
                        # for SMARTLoss, randomly pick sent1 or sent2 to save compute on STS SMART
                        chosen_id, chosen_mask, chosen_emb = random.choice([
                            (b_ids1, b_mask1, sent_embeds1), 
                            (b_ids2, b_mask2, sent_embeds2)
                        ])
                        smart_loss_sts = smart_reg(chosen_id, chosen_mask, 'sts', chosen_emb.detach().clone())
                        
                        # for BPPO
                        # bppo_loss_sts = bpp_opt(chosen_id, chosen_mask, 'sts', chosen_emb)
                        
                        # don't use += here, to avoid inplace operation error
                        loss_sts_sum = loss_sts + smart_loss_sts #+ bppo_loss_sts
                        
            loss = loss_sst + loss_para + loss_sts_sum
            cagrad.step([loss_sst, loss_para, loss_sts_sum], probs)
            
            # update target model (only after weights are updated)
            # bpp_opt.update_target_model()
            
            train_loss += loss.item()
            num_batches += 1
            
        train_loss = train_loss / num_batches
        
        # free memory
        del loss, loss_sst, loss_para, loss_sts, smart_loss_sts, loss_sts_sum
        torch.cuda.empty_cache()
        
        with autocast(device_type='cuda', dtype=amp_dtype):
            dev_sst, *_ = model_eval_sst(sst_dev_dataloader, model, device)
            dev_para = model_eval_para(para_dev_dataloader, model, device)
            dev_sts = model_eval_sts(sts_dev_dataloader, model, device)
        dev_acc = dev_sst + dev_para + dev_sts

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
            patience = 0
        else:
            patience += 1

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev sa :: {dev_sst :.3f}, dev para :: {dev_para :.3f}, dev sts :: {dev_sts :.3f}")

        if patience == args.patience:
            print('Early Stopping!')
            break            
    
    # merge dora and linear layers                
    if args.use_dora:
        # load the saved best model (with dora)
        saved = torch.load(args.filepath, weights_only=False)
        config = saved['model_config']
        model = MultitaskBERT(config)
        apply_dora_to_all(model.bert, rank=args.rank, alpha=args.alpha)
        model.load_state_dict(saved['model'])
        
        # merge and unload dora
        merge_and_unload_all(model.bert)
        
        # save the weights-merged model
        saved['model'] = model.state_dict()
        torch.save(saved, args.filepath)
        print(f'save the merged model to {args.filepath}')
        

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath, weights_only=False)
        config = saved['model_config']

        model = MultitaskBERT(config)
        # if args.use_dora:
        #     apply_dora_to_all(model.bert, rank=args.rank, alpha=args.alpha)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels, para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test, args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels, para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev, args.para_dev, args.sts_dev, split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        # para_test_data = SentencePairTestDataset(para_test_data, args)
        # para_dev_data = SentencePairDataset(para_dev_data, args)
        para_test_data = SentencePairCatTestDataset(para_test_data, args)
        para_dev_data = SentencePairCatDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy, dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, device)
        
        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, device)

        with open(args.sst_dev_out, "w+") as f:
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    # sentiment analysis
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    # paraphrase detection
    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    # semantic textual similarity
    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
    
    parser.add_argument("--use_dora", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--cagrad_c", type=float, default=0.5, help="c for CAGrad")
    parser.add_argument("--max_decay_rate", type=float, default=0.8, help="max decay rate for Annealed Sampling")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--savepath", type=str, default="")
    parser.add_argument("--loadpath", type=str, default="")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if args.savepath:
        args.filepath = args.savepath
    else:
        args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.

    print(args)  # show hyperparameters
    print()
    
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    with autocast(device_type='cuda', dtype=amp_dtype):
        test_multitask(args)
