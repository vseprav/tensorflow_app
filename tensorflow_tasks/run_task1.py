import tensorflow as tf

cluster = tf.train.ClusterSpec({"worker": ["localhost:2220", "localhost:2221"]})

tf.train.Server(cluster, job_name="worker", task_index=0)
tf.train.Server(cluster, job_name="worker", task_index=1)

with tf.Session("grpc://localhost:2220") as sess:
    with tf.device("/job:worker/task:0"):
        const_0_1 = tf.constant("Hello I am the first constant task 0")
        const_0_2 = tf.constant("Hello I am the second constant task 0")

    print(sess.run([const_0_1, const_0_2]))

with tf.Session("grpc://localhost:2221") as sess:
    with tf.device("/job:worker/task:1"):
        const_1_1 = tf.constant("Hello I am the first constant task 1")
        const_2_1 = tf.constant("Hello I am the second constant task 1")

    print(sess.run([const_1_1, const_2_1]))
