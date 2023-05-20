## Fine tuning BERT-base model using SQUAD 2.0 for Question-Answering

In this directory, We have built an an finetuned BERT model to Which can answer a question given a context.

The notebooks are done in order mentioned by the file names.

DATA:
- Downloaded SQUAD 2.0 is stored in SQUAD2.0_DATA folder
- The above folder also contains the transformed SQUAD 2.0 data

We have used Hugging face implementation of BERT base.

We have compared our fine tuned model with pre-trained model.

Training and evaluation takes a lot of time, We used colab pro for implementation.

Summary:

evaluation metrics:
EM: Compare if predicted answer matches with ground truth. 1 if true, 0 otherwise.
F-1: Harmonic mean of precision and recall.

Hyperparameters considered:

Learning Rate: 1e-5, 2e-5, 3e-5, 5e-5
Batch Sie: 4, 6, 8, 10
epochs: 5

