import cv2
import tensorflow as tf
import timeit
import numpy as np
import os

cv2.namedWindow("preview")
vc = cv2.VideoCapture(0)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False

#Define checkpoint directory and inference op
#check_dir = "/media/hunter/1E52113152110F61/shared/Training_Experiments/CycleGAN/horses2zebras/checkpoints"
#check_dir = "/media/hunter/1E52113152110F61/shared/Training_Experiments/CycleGAN/maps2sat/checkpoints"
check_dir = "/media/hunter/1E52113152110F61/shared/Training_Experiments/CycleGAN/cityscapes/checkpoints"
#For CycleGAN A to B
pred_name = 'Model/g_A/t1:0'
data_name = "input_A:0"
checkpoint_file= tf.train.latest_checkpoint(check_dir)
#For CycleGAN A to B
#pred_name = 'Model/g_B/t1:0'
#data_name = "input_B:0"

os.environ["CUDA_VISIBLE_DEVICES"]="1"

#Define image pre-processing function
def pre_process_im(im):
    im = cv2.resize(im,(256,256))
    im = (im.reshape(1, 256, 256, 3).astype(np.float32) / 127.5) - 1
    return im

#Define prediction processing fucntion
def un_process_pred(pred):
    pred = np.squeeze(pred)
    pred = (pred + 1) * 127.5
    pred = pred.astype(np.uint8)
    return pred

with tf.Session() as sess:

    #Restore model and get inference op
    saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
    saver.restore(sess, checkpoint_file)
    infer = tf.get_default_graph().get_tensor_by_name(pred_name)
    input = tf.get_default_graph().get_tensor_by_name(data_name)

    pred_window = "prediction"
    cv2.namedWindow(pred_window,cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(pred_window,600,600)

    in_window = "input"
    cv2.namedWindow(in_window,cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(in_window,600,600)


    while rval:

        #Pull image from webcam and show
        cv2.imshow(in_window, frame)
        rval, frame = vc.read()



        #Run inference on it
        im = pre_process_im(frame)
        pred = sess.run(infer,feed_dict={input:im})

        pred = un_process_pred(pred)

        cv2.imshow(pred_window,pred)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

vc.release()
cv2.destroyWindow(in_window)
cv2.destroyWindow(pred_window)