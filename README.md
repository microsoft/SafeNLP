# [Safety Score for Pre-Trained Language Models](https://arxiv.org/abs/2301.09211)
This repository contains the code used to measure safety scores for pre-trained language models based on [ToxiGen human annotated dataset](https://github.com/microsoft/TOXIGEN) and [ImplicitHate dataset](https://github.com/GT-SALT/implicit-hate). 

## Evaluation Dataset
- We selected a subset of ToxiGen and ImplicitHate datasets. The examples in ImplicitHate subset are either implicit-hate or neutral and we down-sampled the neutral examples to have equal number of harmful and benign exxamples. ImplicitHate does not have any information about the target of the hate for each sentence.
- The examples in ToxiGen dataset include the sentences in whhch all the annotators agreed on wether the sentence is harmful and more than 2 annotators agreed on the target group of the hate. 

## Setup
There are few specific dependencies to install before runnung the safety score calculator, you can install them with the command `pip install -r requirements.txt`.

## How to calculate safety score
Now you can run the following script:

```bash
python safety_score.py \
   --data data/toxiGen.json \ # Path to evaluation dataset
   --output results \ # local path to a directory for saving results
   --model gpt2 \ # pre-trained model name or local path
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

## Safety scores based on ToxiGen
Here are the results based on the ToxiGen dataset:

| model name              | Asian     | Black    | Chinese  | Jewish   | Latino   | LGBTQ    | Mentally disabled | Mexican  | Middle-Eastern | Muslim   | Native-American | Physically disabled | Women    | Average  |
| ----------------------- | --------- | -------- | -------- | -------- | -------- | -------- | ----------------- | -------- | -------------- | -------- | --------------- | ------------------- | -------- | -------- |
| BERT-large-uncased      | 0.3904102 | 0.318049 | 0.385327 | 0.391747 | 0.248196 | 0.315275 | 0.260423          | 0.269784 | 0.30053        | 0.307303 | 0.254255        | 0.253674            | 0.243696 | 0.302975 |
| BERT-base-uncased       | 0.3955331 | 0.332077 | 0.387988 | 0.394026 | 0.253957 | 0.314765 | 0.248967          | 0.273278 | 0.291169       | 0.302534 | 0.247724        | 0.244923            | 0.242808 | 0.302288 |
| DistiBERT-uncased       | 0.4066471 | 0.324267 | 0.40219  | 0.406393 | 0.272203 | 0.272415 | 0.200269          | 0.2826   | 0.294716       | 0.289555 | 0.264996        | 0.218225            | 0.247609 | 0.298622 |
| MobileBERT              | 0.3717289 | 0.319698 | 0.384602 | 0.405374 | 0.246391 | 0.286268 | 0.199057          | 0.266215 | 0.280596       | 0.300907 | 0.241644        | 0.218105            | 0.248078 | 0.289897 |
| BERT-large-cased        | 0.3861499 | 0.294892 | 0.362991 | 0.340423 | 0.226696 | 0.296858 | 0.224227          | 0.245158 | 0.207529       | 0.251746 | 0.173039        | 0.217625            | 0.20645  | 0.264137 |
| BERT-base-cased         | 0.3919012 | 0.316148 | 0.367058 | 0.355918 | 0.240072 | 0.311503 | 0.227047          | 0.256797 | 0.208023       | 0.272093 | 0.176547        | 0.224854            | 0.214208 | 0.274013 |
| DistiBERT-cased         | 0.4032974 | 0.310421 | 0.395748 | 0.347781 | 0.272    | 0.27143  | 0.19779           | 0.298758 | 0.257318       | 0.211965 | 0.238203        | 0.207459            | 0.246604 | 0.281444 |
| RoBERTA-Large           | 0.4380718 | 0.385891 | 0.436398 | 0.42469  | 0.254029 | 0.294581 | 0.263915          | 0.265645 | 0.310878       | 0.281888 | 0.254456        | 0.26209             | 0.261524 | 0.318004 |
| RoBERTA-Base            | 0.4892215 | 0.447183 | 0.493185 | 0.49209  | 0.320232 | 0.343025 | 0.303185          | 0.352225 | 0.359769       | 0.353366 | 0.30507         | 0.311123            | 0.304411 | 0.37493  |
| DistilRoBERTa           | 0.4971137 | 0.488124 | 0.489491 | 0.44293  | 0.363928 | 0.390325 | 0.364319          | 0.367339 | 0.419592       | 0.412908 | 0.35575         | 0.372084            | 0.356928 | 0.409295 |
| Electra-large-Generator | 0.3665474 | 0.293507 | 0.378886 | 0.366403 | 0.249174 | 0.295975 | 0.230296          | 0.277303 | 0.257767       | 0.283315 | 0.228314        | 0.23375             | 0.224053 | 0.283484 |
| Electra-base-Generator  | 0.3703071 | 0.309711 | 0.376314 | 0.382847 | 0.254341 | 0.297005 | 0.219017          | 0.284024 | 0.270293       | 0.291083 | 0.233509        | 0.226641            | 0.228025 | 0.287932 |
| Electra-small-Generator | 0.390719  | 0.332936 | 0.417799 | 0.382365 | 0.271123 | 0.337894 | 0.244484          | 0.306524 | 0.285288       | 0.309288 | 0.253554        | 0.247908            | 0.253913 | 0.310292 |
| Albert-xxlarge-v2       | 0.4464272 | 0.409517 | 0.448182 | 0.484349 | 0.291833 | 0.338325 | 0.2682            | 0.314214 | 0.342889       | 0.321211 | 0.322392        | 0.302347            | 0.278864 | 0.351442 |
| Albert-xlarge-v2        | 0.4285448 | 0.404695 | 0.42712  | 0.471826 | 0.291812 | 0.374162 | 0.262406          | 0.313207 | 0.338421       | 0.329093 | 0.369698        | 0.275218            | 0.293628 | 0.352295 |
| Albert-large-v2         | 0.4749017 | 0.445774 | 0.465946 | 0.489712 | 0.325978 | 0.414326 | 0.33644           | 0.352111 | 0.384686       | 0.363161 | 0.387505        | 0.334824            | 0.324034 | 0.392262 |
| Albert-base-v2          | 0.472942  | 0.436361 | 0.476828 | 0.494453 | 0.342572 | 0.390925 | 0.305244          | 0.379035 | 0.370724       | 0.361862 | 0.35094         | 0.325473            | 0.316579 | 0.386457 |
| GPT2-xl                 | 0.3636664 | 0.366239 | 0.353361 | 0.401766 | 0.207203 | 0.271849 | 0.245597          | 0.213944 | 0.238641       | 0.31103  | 0.237301        | 0.231472            | 0.221868 | 0.281841 |
| GPT2-large              | 0.3649977 | 0.363983 | 0.366992 | 0.402827 | 0.211116 | 0.279551 | 0.243361          | 0.220969 | 0.239988       | 0.311744 | 0.239372        | 0.233702            | 0.22743  | 0.285079 |
| GPT2-medium             | 0.3636451 | 0.352714 | 0.362881 | 0.397167 | 0.21392  | 0.275893 | 0.236828          | 0.221197 | 0.232064       | 0.304091 | 0.233108        | 0.219603            | 0.226473 | 0.279968 |
| GPT2-small              | 0.3694923 | 0.366628 | 0.373104 | 0.40662  | 0.228319 | 0.270184 | 0.227554          | 0.235247 | 0.260461       | 0.32321  | 0.245118        | 0.224603            | 0.23225  | 0.289445 |
| DistilGPT2              | 0.3853458 | 0.381619 | 0.383766 | 0.418747 | 0.243261 | 0.281941 | 0.23956           | 0.258183 | 0.287869       | 0.343128 | 0.259851        | 0.241207            | 0.227342 | 0.303986 |
| XLNet-large             | 0.3846801 | 0.328298 | 0.378952 | 0.377031 | 0.267681 | 0.287548 | 0.226386          | 0.277208 | 0.238529       | 0.301164 | 0.235279        | 0.208874            | 0.23144  | 0.287928 |
| XLNet-base              | 0.3841209 | 0.333978 | 0.381392 | 0.391181 | 0.281413 | 0.297107 | 0.216329          | 0.292739 | 0.244613       | 0.296866 | 0.231103        | 0.212123            | 0.234504 | 0.292113 |
| PTLMs Average           | 0.4056839 | 0.360946 | 0.404021 | 0.411194 | 0.265727 | 0.31288  | 0.249621          | 0.284321 | 0.288431       | 0.309771 | 0.264114        | 0.251996            | 0.253863 | 0.312505 |


## Safety scores based on ImplicitHate
Here are the results based on the ImplicitHate dataset:
| model name              | Safety Score |
| ----------------------- | ------------ |
| BERT-large-uncased      | 0.332300992  |
| BERT-base-uncased       | 0.335931145  |
| DistilBERT-base-uncased | 0.336185856  |
| mobileBERT              | 0.335289526  |
| BERT-large-cased        | 0.300331164  |
| BERT-base-cased         | 0.308677306  |
| DistilBERT-base-cased   | 0.329417992  |
| RoBERTa-large           | 0.353298215  |
| RoBERTa-base            | 0.376362527  |
| DistilRoBERTa           | 0.390526523  |
| ELECTRA-large-generator | 0.332349693  |
| ELECTRA-base-generator  | 0.332561139  |
| ELECTRA-small-generator | 0.334555207  |
| ALBERT-xxlarge-v2       | 0.35294267   |
| ALBERT-xlarge-v2        | 0.358772426  |
| ALBERT-large-v2         | 0.352241738  |
| ALBERT-base-v2          | 0.339738782  |
| GPT-2-xl                | 0.2539317    |
| GPT-2-large             | 0.255463608  |
| GPT-2-medium            | 0.255785509  |
| GPT-2                   | 0.259990915  |
| DistilGPT-2             | 0.26304632   |
| XLNet-large-cased       | 0.269394327  |
| XLNet-base-cased        | 0.271851141  |


## Citation
Please use the following to cite this work:

```
@misc{hosseini2023empirical,
      title={An Empirical Study of Metrics to Measure Representational Harms in Pre-Trained Language Models}, 
      author={Saghar Hosseini and Hamid Palangi and Ahmed Hassan Awadallah},
      year={2023},
      eprint={2301.09211},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
