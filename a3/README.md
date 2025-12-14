# Assignment 3
* [Assignment2].
* For detailed executions, see [assignment3.ipynb].

## Note
Some modifications to make the code works.
* In `beam_search_diagnostics.py`, comment out the line `"u": os.getlogin()`.
* In `utils.py`, use `nltk.download('punkt_tab')` instead of `nltk.download('punkt')`.
* In `run.py` and `nmt_model.py`, add `weights_only=False` to torch.load function:
  ```python
  params = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
  ```


## Result
```
Corpus BLEU: 19.959350420896126
```



