# linear regression model using tensorflow
# y = w * X + b
# loss = (Y - Y_predicred) ^ 2
# to improve we can use huber loss which is robust to outliers


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

DATA_FILE = './data/fire_theft.xls'


# define huber loss
def huber_loss(labels, predictions, delta=1.0):
    residual = tf.abs(predictions - labels)
    condition = tf.less(residual, delta)
    small_res = 0.5 * tf.square(residual)
    large_res = delta * residual - 0.5 * tf.square(delta)
    return tf.where(condition, small_res, large_res)


# read data
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# create placeholders
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# create weight and bias
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

# build models
Y_predicted = X * w + b

# create loss
# loss = tf.square(Y - Y_predicted, name='loss')
loss = huber_loss(Y_predicted, Y)
# using gradient descent with learning rate of 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/linearregression', sess.graph)
    for i in range(100):
        total_loss = 0
        for x, y in data:
            _, l = sess.run([optimizer, loss], feed_dict={X: x, Y: y})
            total_loss += l
        print("Epoch{0}:{1}".format(i, total_loss / n_samples))

    writer.close()

    w_value, b_value = sess.run([w, b])

X, Y = data.T[0], data.T[1]
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * w_value + b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
