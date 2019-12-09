#Tensors
import tensorflow as tf
try:
	#Rank 0
	mammal = tf.Variable("Elephant", tf.string)
	ignition = tf.Variable(451, tf.int16)
	floating = tf.Variable(3.14159, tf.float64)
	its_complicated = tf.Variable((12.3, -4.85), tf.complex64)

	#Rank 1
	mystr = tf.Variable(["Hello"], tf.string)
	cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
	first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
	its_very_complicated = tf.Variable([(12.3, -4.85), (7.5, -6.23)], tf.complex64)

	#Rank 2
	mymat = tf.Variable([[7],[11]], tf.int16)
	myxor = tf.Variable([[False, True],[True, False]], tf.bool)
	linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
	squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
	rank_of_squares = tf.rank(squarish_squares)
	mymatC = tf.Variable([[7],[11]], tf.int32)

	#For Image
	my_image = tf.zeros([2,3,2,5])	#batch * height * width * color

	#Slicing
	my_scalar = first_primes[2]
	my_scalar_2 = squarish_squares[1,1]
	my_row_vector = squarish_squares[1]
	my_column_vector = squarish_squares[:,1]

	print(mammal)
	print(ignition)
	print(floating)
	print(its_complicated)
	print(mystr)
	print(cool_numbers)
	print(first_primes)
	print(its_very_complicated)
	print(mymat)
	print(myxor)
	print(linear_squares)
	print(squarish_squares)
	print(rank_of_squares)
	print(mymatC)
	print(my_image)
	print(my_scalar)
	print(my_scalar_2)
	print(my_row_vector)
	print(my_column_vector)
	print("---------------------------------------------------------")

	init=tf.global_variables_initializer()

	with tf.Session() as session:
		session.run(init)
		print(session.run(mammal))
		print(session.run(ignition))
		print(session.run(floating))
		print(session.run(its_complicated))
		print(session.run(mystr))
		print(session.run(cool_numbers))
		print(session.run(first_primes))
		print(session.run(its_very_complicated))
		print(session.run(mymat))
		print(session.run(myxor))
		print(session.run(linear_squares))
		print(session.run(squarish_squares))
		print(session.run(rank_of_squares))
		print(session.run(mymatC))
		print(session.run(my_image))
		print(session.run(my_scalar))
		print(session.run(my_scalar_2))
		print(session.run(my_row_vector))
		print(session.run(my_column_vector))

except:
	print("Can't Execute")