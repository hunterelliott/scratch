import tensorflow as tf
import argparse
import sys
import os
import time
import numpy as np

# ---- input ---- #

parser = argparse.ArgumentParser()

# Flags for defining the tf.train.ClusterSpec
parser.add_argument(
  "--worker_hosts",
  type=str,
  default="",
  help="Comma-separated list of hostname:port pairs"
)

parser.add_argument(
  "--ps_host",
  type=str,
  default="",
  help="Comma-separated list of hostname:port pairs"
)

parser.add_argument(
  "--task_index",
  type=int,
  default=0,
  help="Index of task within the job"
)
parser.add_argument(
  "--gpu",
  type=int,
  default=0,
  help="Device index of GPU to use. -1 is CPU"
)

parser.add_argument(
  "--job_name",
  type=str,
  default="worker",
  help="Name of job to assign process to"
)
parser.add_argument(
  "--data_dir",
  type=str,
  default="/home/hunter/Desktop/TEMP_LOCAL/data/CAMELYON16_MixedForInferenceTest",
  help="Directory with images to perform inference on"
)

FLAGS, unparsed = parser.parse_known_args()


worker_hosts = FLAGS.worker_hosts.split(",")
n_workers = len(worker_hosts)
task_ind = FLAGS.task_index
job_name = FLAGS.job_name


print(worker_hosts)


# ----- init ----- #

#Reads one image from the current file queue
def get_image(file_queue):

  
  reader = tf.WholeFileReader()
  key,image_data = reader.read(file_queue)
  #image_data = tf.read_file(file_name)
  image = tf.image.decode_image(image_data) 



  return image

file_list = os.listdir(FLAGS.data_dir)
nFiles = len(file_list)

for i in range(nFiles):
  file_list[i] = os.path.join(FLAGS.data_dir, file_list[i])

batch_size = 32
shard_size = batch_size*100 #Avoid excessive communications overhead and returns to python by sharding the samples

# ---- create server, populate queue ----- #

# if job_name == "ps":
#   #Keep the chief on CPU
#   os.environ["CUDA_VISIBLE_DEVICES"]=''  

cluster = tf.train.ClusterSpec({"ps": [FLAGS.ps_host], "worker":worker_hosts})
server = tf.train.Server(cluster, job_name=job_name, task_index=task_ind)

#sample_ind = range(nFiles)
sample_ind = np.arange(0,nFiles)

with tf.device("/job:ps/task:0/cpu:0"):


  #Initialize the file queues on the parameter server
  queue = tf.train.input_producer(sample_ind,capacity=batch_size*10,shared_name=
    'shared_sample_queue',num_epochs=1,shuffle=False)
  
  #image = get_image(queue)
  get_ind = queue.dequeue_many(shard_size)
  #images = tf.train.batch([image],batch_size,num_threads=4,capacity=batch_size*6,shapes=[256,256,3],shared_name='shared_batch_queue')


sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth=True    #TEMP - for testing minimize memory use


coord = tf.train.Coordinator()


if job_name == "ps":

  print("Starting chief process...")
  
  with tf.device("/job:" + job_name + "/task:" + str(task_ind) + "/cpu:0"):
  
    sess = tf.Session(server.target,config=sess_config)
    sess.run(tf.local_variables_initializer()) #Queue runners need local variables initialized
    
    
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)

    server.join()

elif job_name == "worker":


  with tf.device("/job:" + job_name + "/task:" + str(task_ind)):


    print("Starting worker process " + str(task_ind) + "...")    

    sess = tf.Session(server.target,config=sess_config)        
    

    # -- process the images!

    print("Starting inference...")
    start_time = time.time()
    # try:
    
    nIm_proc = 0    

    #Get a batch of sample indices from the shared queue for processing
    inds = sess.run(get_ind)

    print("Processing samples " + str(inds[0]) + " to " + str(inds[-1]))

    #Setup local queue ops for processing
    shard_list = [file_list[i] for i in inds]      
    shard_queue = tf.train.string_input_producer(shard_list,capacity=shard_size,num_epochs=1,shuffle=False)
    image = get_image(shard_queue)
    images = tf.train.batch([image],batch_size,num_threads=4,capacity=batch_size*3,shapes=[256,256,3])


    sess.run(tf.local_variables_initializer()) #Queue runners need local variables initialized
    threads = tf.train.start_queue_runners(coord=coord,sess=sess)

    print("Shard queue configured")

      #threads = tf.train.start_queue_runners(coord=local_coord,sess=sess)
      

    try:

      while not(coord.should_stop()):

        ims = sess.run(images)

        #im = sess.run(get_image(file_list[ind]))
        #qs = sess.run(shard_queue.size())
        #print(qs)
        #Run inference on them
        #pred = sess.run(predict)


        nIm_proc += ims.shape[0]
      #nIm_proc += 1

        if nIm_proc%(2*batch_size) == 0:
          print("Finished " + str(nIm_proc) + " images")

    except tf.errors.OutOfRangeError:
      print("Shard queue exhausted, processed " + str(nIm_proc) + " samples total")

    #   print("Queue exhausted. Completed " + str(nIm_proc) + " images total")

    #finally:
      #coord.request_stop()

    end_time = time.time()
    print("Done, took " + str(end_time-start_time) + " seconds")






    














