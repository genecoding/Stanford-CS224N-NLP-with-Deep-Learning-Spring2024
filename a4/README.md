# Assignment 4
* [Assignment4].
* For detailed executions, see [assignment4.ipynb].

## Note
In `attention.py`, use `theta_i = 10000^(-2(i-1)/dim)` instead of `theta_i = 1/10000^(-2(i-1)/dim)`.

## Result
* Learning curves  
  (x-axis: iterations)  
  <img src="lr curves/train_loss.png" width="50%" />
* Accuracy on the dev set
  ```
  # without pretraining
  Correct: 9.0 out of 500.0: 1.7999999999999998%

  # with pretraining, vinilla position embeddings
  Correct: 115.0 out of 500.0: 23.0%

  # with pretraining, RoPE
  Correct: 202.0 out of 500.0: 40.400000000000006%
  ```

## Reference
[RoFormer: Enhanced Transformer with Rotary Position Embedding], J Su *et al.*


[Assignment4]: https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/assignments/a4_spr24_student_handout.pdf
[assignment4.ipynb]: assignment4.ipynb
[RoFormer: Enhanced Transformer with Rotary Position Embedding]: https://arxiv.org/pdf/2104.09864
