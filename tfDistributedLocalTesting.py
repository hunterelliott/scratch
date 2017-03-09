import tensorflow as tf
import argparse
import sys


# ---- input

parser = argparse.ArgumentParser()
parser.register("type", "bool", lambda v: v.lower() == "true")

# Flags for defining the tf.train.ClusterSpec
parser.add_argument(
  "--worker_hosts",
  type=str,
  default="",
  help="Comma-separated list of hostname:port pairs"
)
# Flags for defining the tf.train.Server
parser.add_argument(
  "--task_index",
  type=int,
  default=0,
  help="Index of task within the job"
)
FLAGS, unparsed = parser.parse_known_args()


worker_hosts = FLAGS.worker_hosts.split(",")
task_ind = FLAGS.task_index


print(worker_hosts)


# ---- create server, execute tasks

cluster = tf.train.ClusterSpec({"local": worker_hosts})
server = tf.train.Server(cluster, job_name="local", task_index=task_ind)


with tf.device("/job:local/task:" + str(task_ind)):

	c = tf.random_normal((1,1000), mean=task_ind, stddev=1.0, dtype=tf.float32, seed=None, name=None)

	sess=tf.Session(server.target)

	cEv = sess.run(c)

	print(cEv)
	print(task_ind)









