# modified by:
#   Steeve LAQUITAINE
#
# description:
#   This is the app's entry point.
#
#   usage:
#       python models.py --pipeline train
#       python models.py --pipeline predict
#       python models.py --pipeline predict_stream

import tensorflow as tf

from src.slSlotRefine.nodes.model import NatSLU
from src.slSlotRefine.nodes.utils import get_catalog, get_params
from src.slSlotRefine.pipelines.predict import (run_predict_pipeline,
                                                run_predict_stream_pipeline)
from src.slSlotRefine.pipelines.train import run_train_pipeline

# set seed for reproducibility
tf.random.set_random_seed(0)


if __name__ == "__main__":

    """Entry point

    usage:
        python models.py --pipeline train
        python models.py --pipeline predict
        python models.py --pipeline predict_stream
    """
    # get params and data catalog
    # note: to view args do print(args)
    args = get_params()
    CATALOG = get_catalog()

    # choose pipeline to run
    if args.pipeline == "train":
        # train
        run_train_pipeline(NatSLU, args, CATALOG)
    elif args.pipeline == "predict":
        # predict
        run_predict_pipeline(NatSLU, args, CATALOG)
    elif args.pipeline == "predict_stream":
        # predict stream
        # utterance = input()
        run_predict_stream_pipeline(NatSLU, args, CATALOG)
    else:
        raise ValueError("Pipeline must be either 'train' or 'predict'")
