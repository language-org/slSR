# modified by:
#   Steeve LAQUITAINE
# description:
#   this is a module that contains all model classes

from os.path import exists

import tensorflow as tf
from src.slSlotRefine.nodes.abstract_models import Model


class NatSLU(Model):
    """SlotRefine model
    """

    def __init__(self, args, catalog):

        # inherit from parent
        super().__init__(args, catalog)

    def _load_from_checkpoint(self, sess):
        """Load model from checkpoint if exists

        Args:
            sess ([type]): tensorflow session
        """
        self.saver = tf.train.Saver(tf.all_variables())
        if exists("./model/checkpoints/") and self.arg.restore:
            self.saver.restore(sess, "./model/checkpoints/model")

    def fit(self, sess):
        """Train and evaluate model"""

        # load model's checkpoint if exists
        self._load_from_checkpoint(sess)

        # loop over epochs, train, eval and save inference
        for epoch in range(self.arg.max_epochs):
            self.logger.info("Epoch: {}".format(epoch))
            self.train_one_epoch(sess, epoch)
            self.evaluation(sess)

    def predict(self, sess):
        """Calculate & write predictions

        Args:
            sess ([type]): tensorflow session
        """
        # load model's checkpoint if exists
        self._load_from_checkpoint(sess)

        # write predictions
        if self.arg.dump:
            self.inference(sess, "", self.arg.remain_diff, self.arg.dump)
