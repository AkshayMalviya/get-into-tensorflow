

#constant
import tensorflow as tf
z=tf.constant(5.2, dtype=tf.float64, name='a')
print(z)

#variable
k=tf.Variable(tf.zeros([2]), name='k')
print(k)
#k=tf.Variable(tf.add(a,b), trianable=False)
#print(k)

#Session
import tensorflow as tf
x=tf.constant(-2.0, name='x', dtype=tf.float32)
a=tf.constant(5.0, name='a', dtype=tf.float32)
b=tf.constant(13.0, name='b', dtype=tf.float32)

y=tf.Variable(tf.add(tf.multiply(a,x),b))

init=tf.global_variables_initializer()

with tf.Session() as session:
	session.run(init)
	print(session.run(y))


#placeholder
import tensorflow as tf
x= tf.placeholder(tf.float32, name='x')
y= tf.placeholder(tf.float32, name='y')

z=tf.multiply(x,y, name='z')

with tf.Session() as session:
	print(session.run(z, feed_dict={x: 2.1, y: 3.0}))