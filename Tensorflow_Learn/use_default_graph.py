# use the default graph to do some computering


import tensorflow as tf

x = 2
y = 3

# actually we should use x=tf.constant(2)
op1 = tf.add(x, y)
op2 = tf.multiply(x, y)
op3 = tf.pow(op2, op1)
useless = tf.multiply(x, op1)
with tf.Session() as sess:
    op3, not_useless = sess.run([op3, useless])
    print(op3, not_useless)
