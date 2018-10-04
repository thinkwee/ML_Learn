import tensorflow as tf

# tf.constant(value,dtype=None,shape=None,name='Const',verify_shape=False)
# if verify_shape=True,it will throw a error when the value and shape you type in don't match

# a=tf.constant(2,shape=[2,2],verify_shape=True)
# it will throw a error
# a=tf.constant(2,shape=[2,2],verify_shape=False)
# it will use the shape and repeat the last value to fulfill the shape

# it will add a vector and matrix since it add the vector to the every dimension of the matrix
# and it also multiply the vector to each dimension of the matrix
# a = tf.constant([2, 2], name="a")
# b = tf.constant([[0, 1], [2, 3]], name="b")
# x = tf.add(a, b, name="add")
# y = tf.multiply(a, b, name="mul")


# tf.zeros(shape,dtype=tf.float32,name=None)
# also tf.ones
# m = tf.zeros([2, 3], tf.int32)

# create a tensor according to the known tensor
# also tf.ones_like
# the tensor must be dense
# input_tensor = [[1, 1, 1], [2, 2, 2]]
# m = tf.zeros_like(input_tensor)
# n = tf.ones_like(input_tensor)


# tf.fill(dims,values,name=None)
# x = tf.fill([2, 3], 8)

# tf.linspace(start,stop,num,name=None)
# not like numpy the start and stop must be float
# x = tf.linspace(10.0, 15.0, 6)

# tf.range(start,limit=None,delta=1,dtype=None,name='range')
# unlike lispace you don't get to the limit
# x=tf.range(3,18,3)
# x=tf.range(1,5)
# x=tf.range(10)

# some random functions
# tf.random_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# tf.truncated_normal(shape,mean=0.0,stddev=1.0,dtype=tf.float32,seed=None,name=None)
# tf.random_uniform(shape,minval=0,maxval=None,dtype=tf.float32,seed=None,name=None)
# tf.random_shuffle(value,seed=None,name=None)
# tf.random_crop(value,size,seed=None,name=None)
# tf.multinomial(logits,num_samples,seed=None,name=None)
# tf.random_gamma(shape,alpha,beta=None,dtype=tf.float32,seed=None,name=None)
# normalrandom
# runcated:more dense normal
# uniformrandom
# shuffle with the first dimension of the tensor
# random crop the image
# logits means the unnormalize probability,given the logits,tf.multinomial() returns a multinomial distribution
# gamma distributiond
# l = tf.random_normal([1, 3])
# x = tf.multinomial(l, 10)
# better set the random seed:   tf.set_random_seed(seed)


# more op
# element-wise math:add,sub,multiply,div,exp,log,greater,less,equal
# array op:concat,slice,split,constant,rank,shape,shuffle
# matrix op:matmul,matrixinverse,matrixdeterminant
# stateful op:variable,assign,assignadd
# neural network building blocks:softmax,sigmoid,relu,convolution2d,maxpool
# checkpointing op:save restore
# queue and synchronization op:enqueue,dequeue,mutexacuire,mutexrelease
# flow control:merge,switch,enter,leave,nextiteraion

# matrix mul must follow the matrix size
# a = tf.constant([3, 6])
# b = tf.constant([2, 2])
# x = tf.matmul(tf.reshape(a, [1, 2]), tf.reshape(b, [2, 1]))

# tensorflow takes python natives types:boolean,numeric(int,float),strings
# also support the op on the tensor of strings,tensors of boolean(zero s_like make all False and ones_like make all True)

# s_1=['apple','banana','peach']
# x=tf.zeros_like(s_1)    #if ones_like it will throw a error

# on datatypes,tensorflow seamlessly integrate with numpy
# but do not use python native types for tensors because tensorflow has to infer python type

# print graph
# from the example below we can see the constant is saved in the graph
# use variable and reader to load real data

# my_const = tf.constant([1.0, 2.0], name="my_const")
# with tf.Session() as sess:
#     print(sess.graph.as_graph_def())


# variables
# constant is just an op ,but Variables is a class
# a = tf.Variable(2, name="scalar")
# b = tf.Variable([2, 3], name="vector")
# c = tf.Variable([[0, 1], [2, 3]], name="matrix")
# w = tf.Variable(tf.zeros([100, 10]))

# variables op
# x=tf.Variable()
# x.initializer run it with sess it will initialize single variable
# x.value()
# x.assign()
# x.assign_add() x.assign_sub()

# constant values are stored in the graph definition
# Sessions allocate memory to store variable values


# init all variables at once
# init=tf.global_variables_initializer()
# or just initialize only a subset of variables
# init_ab=tf.variables_initializer([a,b],name="init_ab")


# in session you can just print(a.eval()),but first you should run w.initializer
# for a.eval() tensorflow will create a default session and run it

# the assign can be assigned to an op
# actually the initializaer is one kind of assign_op
# assign_op=a.assign(100)
# sess.run(assign_op)

# if you use a.assign(),you should run it. Just run a.initializer won't assign the value

# each session has its own copy of variable

# w=tf.Variable(10)
# sess1=tf.Session()
# sess2=tf.Session()
#
# sess1.run(w.initializer)
# sess2.run(w.initializer)
#
# print(sess1.run(w.assign_add(10)))
# print(sess2.run(w.assign_sub(2)))
#
# sess1.close()
# sess2.close()

# you can use a variable to initialize another variable,quite common but not safe cause w may be not safe

# w=tf.Variable(tf.truncated_normal([10,10]))
# u=tf.Variable(2*w)
# we should use the safe way
# u=tf.Variable(2*w.initialized_value())


# Session vs InteractiveSession
# the only difference is an InteractiveSession makes itself the default
# we can use c.eval() in InteractiveSession
# which is tf.get_default_session().run(c).eval() in Session


# tf.Graph.control_dependencies(control_inputs)
# it defines which ops should be run first
# such as d and e will only run after a,b,c have executed
#
# with g.control_dependencies([a,b,c]):
#     d=
#     e=

# placeholders
# assemble the graph first without knowing the values needed for computation
# unlike constant and variable,placeholder does not take space in a graph
# tf.placeholder(dtype,shape=None,name=None)

# when you run a session contains placeholders,you should first feed the placeholders with dictionaries
# if shape of the placeholder is None it means the placeholder can take any shape
# in tensorboard,placeholder will be depicted as a op node
# a = tf.placeholder(tf.float32, shape=[3])
# b = tf.constant([5, 5, 5], tf.float32)
# c = a + b
# with tf.Session() as sess:
#     print(sess.run(c, {a: [1, 2, 3]}))

# feed multiple data points in placeholder:
# with tf.Session() as sess:
#     for a_avlue in list_of_values_for_a:
#         print(sess.run(c, {a: a_avlue}))

# besides placeholder,tensor are feedable too.placeholder is just a tensor that must be fed
# tf.Graph.is_feedable(tensor)

# a = tf.add(2, 5)
# b = tf.multiply(a, 3)
#
# with tf.Session() as sess:
#     replace_dict = {a: 15}
#     b = sess.run(b, feed_dict=replace_dict)
#     print(b) #returns 45

# avoid lazy loading which means creating/initializing an ovject until it is needed
# so you should define all the op nodes instead of declaring it in the sess.run()
# otherwise in loop conditions it will create multiple op nodes
# solution:
# seperate definition of ops from computinng/running ops
# use python property to ensure function is also loaded once the first time it is called

# the python property example:make a function into a property:
# we can see when we first call prediction it will do some initialization
# when we call it second time it just return the handle of self._prediction,and the handle has contained the information
# so only one prediction node will be created in the graph
# in a word:use python attribute to ensure a function is only loaded the first time it's called

# class Model:
#     def __init__(self, data, target):
#         self.data = data
#         self.target = target
#         self._prediction = None
#         self._option = None
#         self._error = None
#
#     @property
#     def prediction(self):
#         if not self.prediction:
#             data_size = int(self.data.get_shape()[1])
#             target_size = int(self.target.get_shape()[1])
#             weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
#             bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
#             incoming = tf.matmul(self.data, weight) + bias
#             self.prediction = tf.nn.softmax(incoming)
#         return self._prediction

# normal loading:
# x = tf.Variable(10, name='x')
# y = tf.Variable(20, name='y')
# z = tf.add(x, y)
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter('./graphs/normalloading',sess.graph)
#     for _ in range(3):
#         sess.run(z)
#     writer.close()

# lazy loading which contains 3 add nodes
# x = tf.Variable(10, name='x')
# y = tf.Variable(20, name='y')
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     writer = tf.summary.FileWriter('./graphs/lazyloading', sess.graph)
#     for _ in range(3):
#         sess.run(tf.add(x, y))
#     writer.close()

# with tf.Session() as sess:
#     x = sess.run(x)
#     print(x)
