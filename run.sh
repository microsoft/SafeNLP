# Calculate safety score for GPT2-small on ToxiGen human annotated dataset
python safety_score.py --data data/toxiGen.json --output results --model gpt2 --lmHead clm 

# Calculate safety score for BERT-base-uncased on ToxiGen human annotated dataset
python safety_score.py --data data/toxiGen.json --output results --model bert-base-uncased --lmHead mlm 

# Calculate safety score for GPT2-small on implicitHate dataset
python safety_score.py --data data/implicitHate.json --output results --model gpt2 --lmHead clm 
