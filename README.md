# Safety Score for Pre-Trained Language Models
This repository contains the code used to measure safety scores for pre-trained language models based on [ToxiGen human annotated dataset](https://github.com/microsoft/TOXIGEN) and [ImplicitHate dataset](https://github.com/GT-SALT/implicit-hate). 

## Evaluation Dataset
- We selected a subset of TxiGen and ImplicitHate datasets. The examples in ImplicitHate subset are either implicit-hate or neutral and we down-sampled the neutral examples to have equal number of harmful and benign exxamples. ImplicitHate does not have any information about the target of the hate for each sentence.
- The examples in ToxiGen dataset include the sentences in whhch all the annotators agreed on wether the sentence is harmful and more than 2 annotators agreed on the target group of the hate. 

## Setup
There are few specific dependencies to install before runnung the safety score calculator, you can install them with the command `pip install -r requirements.txt`.

## How to calculate safety score
Now you can run the following script:

```bash
python safety_score.py \
   --data data/toxiGen.json \ # Path to evaluation dataset
   --output results \ # local path to a directory for saving results
   --model gpt2 \ # pre-trained model name or loccal path
   --lmHead clm \ # Type of language model head, i.e. causal or masked
   --force # overwrites the output path if it already exists.
```
