
# modified by: Steeve LAQUITAINE
# This is the app's entry point
#
#   usage:   
#
#       python models.py --pipeline train
#       python models.py --pipeline predict

from os.path import exists

import tensorflow as tf

from src.slSlotRefine.nodes.inference import write_predictions
from src.slSlotRefine.nodes.model import Model
from src.slSlotRefine.nodes.utils import get_catalog, get_params

# set seed for reproducibility
tf.random.set_random_seed(0)


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
            self.logger.info('Epoch: {}'.format(epoch))
            self.train_one_epoch(sess, epoch)
            self.evaluation(sess)

    def predict(self, sess):
        """Create and write predictions

        Args:
            sess ([type]): tensorflow session
        """
        # load model's checkpoint if exists
        self._load_from_checkpoint(sess)

        # write predictions
        if self.arg.dump:
            self.inference(sess, "", self.arg.remain_diff, self.arg.dump)

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
        
        # instantiate config
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        # train model and save
        with tf.Session(config=config) as sess:
            model = NatSLU(args, catalog=CATALOG)
            sess.run(tf.global_variables_initializer())
            model.fit(sess)
            model.save(sess)
        print('Model Trained Successfully!!')

    elif args.pipeline == "predict":
        
        # predict and write predictions
        with tf.Session() as sess:
            model = NatSLU(args, catalog=CATALOG)
            model.predict(sess)
            write_predictions(CATALOG["inference"]["load_path"], CATALOG["inference"]["write_path"])        
        print('Predictions done Successfully!!')

    
