# modified by:
#   Steeve LAQUITAINE
#
# description:
#   This is the app's entry point.
#
#   usage:
#       python models.py --pipeline train
#       python models.py --pipeline predict

import tensorflow as tf

from src.slSlotRefine.nodes.model import NatSLU
from src.slSlotRefine.nodes.utils import get_catalog, get_params
from src.slSlotRefine.pipelines.predict import run_predict_pipeline
from src.slSlotRefine.pipelines.train import run_train_pipeline

# set seed for reproducibility
tf.random.set_random_seed(0)


if __name__ == "__main__":

    """Entry point

    usage:
        python model.py --pipeline train ...
    """
    # get params and data catalog
    args = get_params()
    CATALOG = get_catalog()

    # choose pipeline to run
    if args.pipeline == "train":
        run_train_pipeline(NatSLU, args, CATALOG)
    elif args.pipeline == "predict":
        run_predict_pipeline(NatSLU, args, CATALOG)
    else:
        raise ValueError("Pipeline must be either 'train' or 'predict'")
