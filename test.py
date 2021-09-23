import tensorflow as tf

# mock variables to checkpoint and re-load
bias = tf.Variable(2.0, name="bias")
 
# Define a test operation to restore for test
sess = tf.Session()
saver = tf.train.Saver()

# initialize values
sess.run(tf.global_variables_initializer())

# run the operation with some first inputs
print(sess.run(bias)) # prints bias
 
# Now, save the graph
saver.save(sess, './test/test', global_step=1000)

# to check a variable in the graph
graph = tf.get_default_graph()
bias = graph.get_tensor_by_name("bias:0")


# Load model graph and parameters and get the values of stored bias variable
# and multiply operation outpyt
with tf.Session() as sess:    
    
    # load graph
    saver = tf.train.import_meta_graph('./test/test-1000.meta')

    # load parameters
    saver.restore(sess, tf.train.latest_checkpoint('./test/'))

    # print restored bias
    print(sess.run('bias:0'))
