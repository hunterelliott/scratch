import tensorflow as tf
import argparse
import sys
import os


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

parser.add_argument(
  "--data_dir",
  type=str,
  default="/home/hunter/Desktop/TEMP_LOCAL/Data/CAMELYON/MixedForInferenceTest/",
  help="Directory with images to perform inference on"
)

FLAGS, unparsed = parser.parse_known_args()


worker_hosts = FLAGS.worker_hosts.split(",")
task_ind = FLAGS.task_index


print(worker_hosts)


fileList = os.listdir(FLAGS.data_dir)
nFiles = len(fileList)

# ---- create server, execute tasks

cluster = tf.train.ClusterSpec({"local": worker_hosts})
server = tf.train.Server(cluster, job_name="local", task_index=task_ind)

with tf.device("/job:local/task:0"):

	#file_queue = tf.FIFOQueue(5e5,tf.string,shared_name='chief_queue')
  filename_queue = tf.train.string_input_producer(fileList,capacity=nFiles,shared_name=
    'chief_queue',num_epochs=1,shuffle=False)

with tf.device("/job:local/task:" + str(task_ind)):

	

	c = tf.random_normal((1,1000), mean=task_ind, stddev=1.0, dtype=tf.float32, seed=None, name=None)

	sess = tf.Session(server.target)

	cEv = sess.run(c)

	sess.run(filename_queue.size())

	


