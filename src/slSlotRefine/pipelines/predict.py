
import tensorflow as tf
from src.slSlotRefine.nodes.inference import write_predictions


def run_predict_pipeline(model, args, CATALOG):  
    
    # predict & write
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        model = model(args, catalog=CATALOG)
        model.predict(sess)
        write_predictions(CATALOG["inference"]["load_path"], CATALOG["inference"]["write_path"])        
    print('Predictions done Successfully!!')

