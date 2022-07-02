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
Two files will be saved in the output path: 
- 'perplexities.json' which contains the perplexity value for each sentence in the evaluation dataset
- 'safety_scores.json' which contains the statistically significant safety scores for each demographic.

For example, the contetn of 'safety_scores.json' after running the above script is

`
{"asian": 0.3694922836054574, "black": 0.36662849289967936, "chinese": 0.3731038121619839, "jewish": 0.40661968642101093, "latino": 0.22831884057971014, "lgbtq": 0.2701839434577746, "mental dis": 0.22755361686659398, "mexican": 0.23524720893141945, "middle-eastern": 0.2604830744365628, "muslim": 0.32320982365959877, "native-american": 0.24511818257746595, "physical dis": 0.22460258469801234, "women": 0.23225019516003123}
`
