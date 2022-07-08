""" 
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
Utility fuctions 
"""

import argparse
import torch
from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True,
                        help='Path to evaluation dataset. i.e. implicitHate.json or toxiGen.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to result text file')
    parser.add_argument('--model', type=str, required=True,
                        help="a local path to a model or a model tag on HuggignFace hub.")
    parser.add_argument('--lmHead', type=str, required=True,
                        choices=['mlm', 'clm'])
    parser.add_argument('--config', type=str,
                        help='Path to model config file')
    parser.add_argument("--force", action="store_true", 
                        help="Overwrite output path if it already exists.")
    args = parser.parse_args()

    return args


def load_tokenizer_and_model(args, from_tf=False):
    '''
    Load tokenizer and model to evaluate.
    '''

    pretrained_weights = args.model
    if args.config:
        config = AutoConfig.from_pretrained(args.config)
    else:
        config = None
    tokenizer = AutoTokenizer.from_pretrained(pretrained_weights) 
    # Load Masked Language Model Head
    if args.lmHead == 'mlm':
        model = AutoModelForMaskedLM.from_pretrained(pretrained_weights,
                                                     from_tf=from_tf, config=config)        
    # load Causal Language Model Head
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_weights,
                                                     from_tf=from_tf, config=config)

    model = model.eval()
    if torch.cuda.is_available():
        model.to('cuda')

    return tokenizer, model
