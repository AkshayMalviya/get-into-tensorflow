#Broadcasting
import  tensorflow as tf
x = tf.constant([1,2,3,5], name = 'x')
y = tf.constant(9, name = 'y')
z = x + y

a = tf.constant([[1, 2, 3], [4, 5, 6]], name='a')
b = tf.constant(100, name='b')
c = a + b

d = tf.constant([[1, 2, 3], [4, 5, 6]], name='d')
e = tf.constant([100, 101, 102], name='e')
f = d + e

g = tf.constant([[1,2,3],[4,5,6]], name = 'g')
h = tf.constant([[100],[101]], name = 'h')
i = g + h

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(z))
	print(session.run(c))
	print(session.run(f))
	print(session.run(i))