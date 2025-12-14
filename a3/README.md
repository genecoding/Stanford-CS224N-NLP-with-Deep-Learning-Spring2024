# Assignment 3
* [Assignment3].
* For detailed executions, see [assignment3.ipynb].

## Note
Some modifications to make the code works.
* In `beam_search_diagnostics.py`, comment out the line `"u": os.getlogin()`.
* In `utils.py`, use `nltk.download('punkt_tab')` instead of `nltk.download('punkt')`.
* In `run.py` and `nmt_model.py`, add `weights_only=False` to `torch.load` function call, see [here] for more details.

## Result
* Learning curves  
  (x-axis: iterations)  
  * Loss  
    <img src="lr curves/loss_train.png" width="30%" />
    <img src="lr curves/loss_val.png" width="30%" />  
  * Perplexity  
    <img src="lr curves/perplexity_train.png" width="30%" />
    <img src="lr curves/perplexity_val.png" width="30%" />
* BLEU score
  ```
  Corpus BLEU: 19.959350420896126
  ```



[Assignment3]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/assignments/a3_spr24_student_handout.pdf
[assignment3.ipynb]: assignment3.ipynb
[here]: https://docs.pytorch.org/docs/stable/notes/serialization.html#weights-only
