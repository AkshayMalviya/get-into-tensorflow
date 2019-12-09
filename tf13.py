import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 28, 28,1])
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
init = tf.global_variables_initializer()

#model
y = tf.nn.softmax(tf.matmul(tf.reshape(x,[-1, 784]),w)+b)

#placeholder for correct answer
y_ = tf.placeholder(tf.float32, [None, 10])

#loss function
cross_entropy = -tf.reduce_sum(y_ + tf.log(y))

# % of correct answer found in batch
is_correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

sess = tf.Session()
sess.run(init)

for i in range(10000):
	#Load batch of images and correct answer
	batch_x, batch_y = mnist.train.next_batch(100)
	train_data = {x: batch_x, y: batch_y}

	#train 
	sess.run(train_step, feed_dict = train_data)

	#success ? add code to print it
	a, c = sess.run([accuracy, croos_entropy], feed = train_data)

	#success on test data ?
	test_data = {x: mnist.test.images, y_ : mnist.test.labels}
	a, c = sess.run([accuracy, cross_entropy], feed = test_data)