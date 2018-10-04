import tensorflow as tf

# step 1:create the summary

with tf.name_scope("summaries"):
    tf.summary.scalar("loss", self.loss)
    tf.summary.scalar("accuracy", self.accuracy)
    tf.summary.histogram("histogram loss", self.loss)
    # merge them all
    self.summary_op = tf.summary.merge_all()

# step 2:then you can run it
# summary is op too

loss_batch, _, summary = sess.run([model_loss, model.optimizer, model.summary_op], feed_dict=feed_dict)

# step 3: write summaries to file

writer.add_summary(summary, global_step=step)


# also we can write a class of attaching varies summaries to a tensor

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)