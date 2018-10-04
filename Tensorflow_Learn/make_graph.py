import tensorflow as tf

# the actual default graph
x = 2
y = 3
op1 = tf.add(x, y)

# now we change the default graph
g = tf.Graph()
with g.as_default():
    x = tf.add(3, 5)

# output graph g
sess = tf.Session(graph=g)
x = sess.run(x)
print(x)
sess.close()

# g2 should be the global default graph
g2 = tf.get_default_graph()

# test g2
sess2 = tf.Session(graph=g2)
op1 = sess2.run(op1)
print(op1)
sess2.close()

# distinguish the default graph and the user graph
# NO MORE THAN ONE GRAPH!
