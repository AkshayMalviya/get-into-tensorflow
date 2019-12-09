#Segmentation
import tensorflow as tf
import numpy as np

def convert(v, t=tf.float32):
	return tf.convert_to_tensor(v, dtype=t)

seg_ids = tf.constant([0,0,1,2,2])
tens1 = convert(np.array([(2,5,3,-5),(0,3,-2,5),(4,3,5,3),(6,1,4,0),(6,1,4,0)]), tf.int32)
tens2 = convert(np.array([1,2,3,4,5]), tf.int32)

seg_sum = tf.segment_sum(tens1, seg_ids)
seg_sum_1 = tf.segment_sum(tens2, seg_ids)

with tf.Session() as session:
	print("Segmentation sum tens1: \n", session.run(seg_sum))
	print("Segmentation sum tens2: \n", session.run(seg_sum_1))
