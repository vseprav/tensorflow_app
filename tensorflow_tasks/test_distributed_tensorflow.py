import tensorflow as tf

"""
    In-graph replication
    run in command line
    python test_distributed_tensorflow.py --hosts=localhost:3007,localhost:3008 --job_name=local --task_id=0 &
    python test_distributed_tensorflow.py --hosts=localhost:3007,localhost:3008 --job_name=local --task_id=1
"""

tf.app.flags.DEFINE_string('job_name', '', 'One of local worker')
tf.app.flags.DEFINE_string('hosts', '', """Comma-separated list of hostname:port for the """)

tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of local/replica running the training')

FLAGS = tf.app.flags.FLAGS

hosts = FLAGS.hosts.split(',')

cluster = tf.train.ClusterSpec({FLAGS.job_name: hosts})
server = tf.train.Server(cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_id)

task_id = str(FLAGS.task_id)

with tf.device('/job:'+FLAGS.job_name+'/task:' + task_id):
    const = tf.constant('Hello I am the task: ' + task_id)

with tf.Session('grpc://' + hosts[FLAGS.task_id]) as sess:
        print(sess.run(const))
