# modified by: steeve laquitaine

import json
import os
import pprint
from collections import defaultdict

pp = pprint.PrettyPrinter(indent=4)

def write_predictions(load_filepath:str, write_filepath:str):
    """write prediction to disk

    Args:
        load_filepath (str): [description]
        write_filepath (str): [description]
    """
    # get utterance, slots, intents
    with open(load_filepath, encoding="utf-8") as file:
        lines = file.readlines()
        lines = [line.strip().lower().split('\t') for line in lines]
        utterances, intents, slots = zip(*lines)

    # format intent with its slots
    struct_outs = defaultdict(dict)
    for u_i in range(len(utterances)):
        struct_outs[u_i]["utterance"]= utterances[u_i]
        struct_outs[u_i]["struct_intent"] = dict()
        for tok_i, slot in enumerate(slots[u_i].split()):
            tokens = utterances[u_i].split()
            if not slot=="o":
                struct_outs[u_i]["struct_intent"]["intent"] = intents[u_i]
                struct_outs[u_i]["struct_intent"][slot] = tokens[tok_i]

    # write predictions to disk
    with open(write_filepath, "w") as outfile:
        json.dump(struct_outs, outfile, indent = 4)

