import tensorflow as tf
import argparse
import sys
import os


# ---- input

parser = argparse.ArgumentParser()
#parser.register("type", "bool", lambda v: v.lower() == "true")

# Flags for defining the tf.train.ClusterSpec
parser.add_argument(
  "--worker_hosts",
  type=str,
  default="",
  help="Comma-separated list of hostname:port pairs"
)

parser.add_argument(
  "--data_dir",
  type=str,
  default="/home/hunter/Desktop/TEMP_LOCAL/Data/CAMELYON/MixedForInferenceTest/",
  help="Directory with images to perform inference on"
)

FLAGS, unparsed = parser.parse_known_args()

# Chief is always task index 0
task_ind = 0

worker_hosts = FLAGS.worker_hosts.split(",")

print(worker_hosts)


fileList = os.listdir(FLAGS.data_dir)
nFiles = len(fileList)



# ---- create server, populate queue ----- #

cluster = tf.train.ClusterSpec({"local": worker_hosts})
server = tf.train.Server(cluster, job_name="local", task_index=task_ind)


with tf.device("/job:local/task:" + str(task_ind)):

  sess = tf.Session(server.target)
  #sess = tf.InteractiveSession()
  #sess.run(tf.global_variables_initializer())


  #fileListGen = tf.train.match_filenames_once(FLAGS.data_dir + '*.png')
  filename_queue = tf.train.string_input_producer(fileList,capacity=nFiles,shared_name=
    'chief_queue',num_epochs=1,shuffle=False)

  sess.run(tf.local_variables_initializer())

  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord,sess=sess)


#fileList = sess.run(fileListGen)
# #   

# for i in range(nFiles):
#   fileList[i] = sess.run(tf.reshape(fileList[i],[1]))


# q = tf.FIFOQueue(nFiles,tf.string,shared_name='chief_queue')

# #   #Create ops to initialize and pull from the queue
# init = q.enqueue_many(fileList)




# #   #Create ops to initialize and pull from the queue
# init = q.enqueue_many(fileList)

# sess.run(init)
  

# 	c = tf.random_normal((1,1000), mean=task_ind, stddev=1.0, dtype=tf.float32, seed=None, name=None)

# 	sess=tf.Session(server.target)

# 	cEv = sess.run(c)

# 	print(cEv)
# 	print(task_ind)









