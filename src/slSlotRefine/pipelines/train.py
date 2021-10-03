        
import tensorflow as tf


def run_train_pipeline(model, args, CATALOG):       
    """Run training pipeline

    Args:
        model ([type]): [description]
        args ([type]): [description]
        CATALOG ([type]): [description]
    """
    # instantiate config
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    
    # train model and save
    with tf.Session(config=config) as sess:
        model = model(args, catalog=CATALOG)
        sess.run(tf.global_variables_initializer())
        model.fit(sess)
        model.save(sess)
    
    print('(run_train_pipeline) Model Trained Successfully!!')
