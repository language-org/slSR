# coding=utf-8
# Copyright 2017-2019 The THUMT Authors

from __future__ import absolute_import, division, print_function

from src.slSlotRefine.nodes.thumt.models import (rnnsearch, rnnsearch_lrp,
                                                 seq2seq, transformer,
                                                 transformer_lrp)


def get_model(name, lrp=False):
    name = name.lower()

    if name == "rnnsearch":
        if not lrp:
            return rnnsearch.RNNsearch
        else:
            return rnnsearch_lrp.RNNsearchLRP
    elif name == "seq2seq":
        return seq2seq.Seq2Seq
    elif name == "transformer":
        if not lrp:
            return transformer.Transformer
        else:
            return transformer_lrp.TransformerLRP
    else:
        raise LookupError("Unknown model %s" % name)
