import tensorflow as tf

# define model

# create a   saver object
# if you don't provide parameters to Saver it will default save all variables
saver = tf.train.Saver()

# very commoin in tensorflow program
self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# launch a session to compute graph
with tf.Session() as sess:
    # actual training loop
    for step in range(training_steps):
        sess.run([optimizer])

        if (step + 1) % 1000=0:
            saver.save(sess, 'checkpoint_directory/model_name', global_step=model.globalstep)

# some important parameters of saver

# max_to_keep:indicate the max number of recent checkpoint files to max_to_keep,default to 5
# kepp_checkpoint_every_n_hours:save the checkpoint every n hours
