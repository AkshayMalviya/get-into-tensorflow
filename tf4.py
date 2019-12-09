#1D tensor
import tensorflow as tf
import numpy as np
tensor_1d= np.array([1.45,-1,0.2,102.1])
print(tensor_1d)
print(tensor_1d[0])
print(tensor_1d[2])
print(tensor_1d.ndim)
print(tensor_1d.shape)
print(tensor_1d.dtype)

tensor = tf.convert_to_tensor(tensor_1d,dtype=tf.float64)

with tf.Session() as session:
	print(session.run(tensor))
	print(session.run(tensor[0]))
	print(session.run(tensor[1]))


#2D Tensor
import tensorflow as tf
import numpy as np
tensor_2d = np.array(np.random.rand(4,4), dtype='float32')
tensor_2d_1 = np.array(np.random.rand(4,4), dtype='float32')
tensor_2d_2 = np.array(np.random.rand(4,4), dtype='float32')

print(tensor_2d)
print(tensor_2d_1)
print(tensor_2d_2)


m1 = tf.convert_to_tensor(tensor_2d)
m2 = tf.convert_to_tensor(tensor_2d_1)
m3 = tf.convert_to_tensor(tensor_2d_2)

print(m1)
print(m2)
print(m3)

mat_product = tf.matmul(m1,m2)
mat_sum = tf.add(m2,m3)
mat_det = tf.matrix_determinant(m3)

with tf.Session() as session:
	print(session.run(mat_product))
	print(session.run(mat_sum))
	print(session.run(mat_det))

tensor_1d_1= np.array([0,0,0])
tensor = tf.convert_to_tensor(tensor_1d_1, dtype=tf.float64)
with tf.Session() as session:
	print(session.run(tf.cos(tensor)))

