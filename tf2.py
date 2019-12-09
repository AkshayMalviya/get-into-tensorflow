
import tensorflow as tf
my_cost=tf.constant([1.0,2.0],name="my_const")
print(tf.get_default_graph().as_graph_def())



import tensorflow as tf
x = tf.constant([1.0,2,3.4,4], name = 'x')
y = tf.Variable(x+5.1, name = 'y')

model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	print(session.run(y))


import tensorflow as tf

x = tf.constant(5, name = 'x')
print(x)
y = tf.Variable(x + 5, name = 'y')

with tf.Session() as session:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('logs',session.graph)
	model = tf.global_variables_initializer()
	session.run(model)
	print(session.run(y))
	
import tensorflow as tf
a=tf.Variable(5, name='a')
d=tf.get_variable('a',initializer='2')

model= tf.global_variables_initializer()

with tf.Session() as session:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter('logs/new',session.graph)
	session.run(model)
	print(session.run(a))
	print(session.run(d))
