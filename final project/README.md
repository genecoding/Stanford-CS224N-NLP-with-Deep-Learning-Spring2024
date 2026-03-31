# Final Project
* Brief [instructions] for [default final project].
* For detailed executions, see [final_project.ipynb].

## Part 1
### Result
```
# fine-tune-mode='last-linear-layer'
Evaluating on SST...
dev acc :: 0.399
Evaluating on cfimdb...
dev acc :: 0.780

# fine-tune-mode='full-model'
Evaluating on SST...
dev acc :: 0.521
Evaluating on cfimdb...
dev acc :: 0.959
```

## Part 2
### Note
My training plan:
1. Further Pretraining  
   Further pretrain BERT using Masked Language Modeling (MLM) on all training sets.
2. Fine-tuning  
   Apply DoRA to all linear layers in BERT and finetune the MultitaskBERT model on all training sets. Note that when using LoRA/DoRA, there is no difference between setting `fine-tune-mode` to `'last-linear-layer'` or `'full-model'`.

I tried the following extensions:
* Data Preprocessing  
  Removed pairs containing empty sentences in the Quora dataset and duplicate pairs in the STS dataset.
* Data Augmentation via Back Translation  
  For the SST dataset, translated sentences to German and back to English by using Google Translate via Google Sheets and removed duplicates.
* Masked Language Modeling (MLM)  
* LoRA / DoRA + Detaching V_norm  
  Conducted a grid search over rank r (powers of 2 from 8 to 256) with α=r and α=2r across various learning rates and selected the best-performing combination.
* Multiple Negatives Ranking Loss (MNRL) / MNRL with Hard Negatives  
  Utilized label=1 datapoints from the Quora dataset for MNRL, but the results were suboptimal. Constructed triplets from duplicate sentences in the Quora dataset for MNRL with Hard Negatives, but the results were likewise underwhelming. These might be more effective as further pretraining methods.
* CoSENTLoss / AnglELoss  
  AnglELoss is theoretically superior to CoSENTLoss, but CoSENTLoss performed better in this setting. It is possible that DoRA affects AnglELoss performance. Since these losses operate directly on text embeddings, no task head is required for the similarity task.
* PCGrad / CAGrad  
  Performance varied between the two depending on the configuration. CAGrad was ultimately selected.
* Annealed Sampling  
  Used annealed sampling probabilities as task weights.
* SMART  
  Comprising Smoothness-Inducing Adversarial Regularization and Bregman Proximal Point Optimization (BPPO). BPPO yielded no obvious improvement in this context.
* Mean Pooling  
* Bi-Encoding & Cross Encoding  
  Employed cross encoding for the paraphrase task and bi-encoding for the similarity task.
* Sharing Weights  
  Added a shared layer between BERT and the sentiment/paraphrase heads.
* Automatic Mixed Precision (AMP)  

While I experimented with numerous hyperparameters, many remain unexplored. Better configurations may exist, but due to highly constrained computing resources, this concludes my optimization.

### Result
```
# dev set
Sentiment classification accuracy: 0.551
Paraphrase detection accuracy: 0.895
Semantic Textual Similarity correlation: 0.865
```

## Reference
* [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding], J Devlin *et al.*
* [LoRA: Low-Rank Adaptation of Large Language Models], E Hu *et al.*
* [DoRA: Weight-Decomposed Low-Rank Adaptation], SY Liu *et al.*
* [CoSENT: Consistent Sentence Embedding via Similarity Ranking], X Huang *et al.*
* [AoE: Angle-optimized Embeddings for Semantic Textual Similarity], X Li *et al.*
* [Gradient Surgery for Multi-Task Learning], T Yu *et al.*
* [Conflict-Averse Gradient Descent for Multi-task Learning], B Liu *et al.*
* [SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization], H Jiang *et al.*
* [Automatic Mixed Precision examples]



[instructions]: README_.md
[default final project]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf
[final_project.ipynb]: final_project.ipynb
[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding]: https://arxiv.org/pdf/1810.04805
[LoRA: Low-Rank Adaptation of Large Language Models]: https://arxiv.org/pdf/2106.09685
[DoRA: Weight-Decomposed Low-Rank Adaptation]: https://arxiv.org/pdf/2402.09353
[CoSENT: Consistent Sentence Embedding via Similarity Ranking]: https://penghao-bdsc.github.io/papers/CoSENT_TASLP2024.pdf
[AoE: Angle-optimized Embeddings for Semantic Textual Similarity]: https://aclanthology.org/2024.acl-long.101.pdf
[Gradient Surgery for Multi-Task Learning]: https://arxiv.org/pdf/2001.06782
[Conflict-Averse Gradient Descent for Multi-task Learning]: https://arxiv.org/pdf/2110.14048
[SMART: Robust and Efficient Fine-Tuning for Pre-trained Natural Language Models through Principled Regularized Optimization]: https://arxiv.org/pdf/1911.03437
[Automatic Mixed Precision examples]: https://docs.pytorch.org/docs/stable/notes/amp_examples.html
