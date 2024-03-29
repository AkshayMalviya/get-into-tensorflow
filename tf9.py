#KNN Algorithm
import os
import numpy as np
import tensorflow as tf

ccf_train_data = "train_dataset.csv"
ccf_test_data = "test_dataset.csv"

dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'D:/Tensorflow/Practice Code/KNN Dataset'))

ccf_train_filepath = os.path.join(dataset_dir, ccf_train_data)
ccf_test_filepath = os.path.join(dataset_dir, ccf_test_data)

def load_data(filepath):
    from numpy import genfromtxt

    csv_data = genfromtxt(filepath, delimiter=",", skip_header=1)
    data = []
    labels = []

    for d in csv_data:
        data.append(d[:-1])
        labels.append(d[-1])

    return np.array(data), np.array(labels)

train_dataset, train_labels = load_data(ccf_train_filepath)
test_dataset, test_labels = load_data(ccf_test_filepath)

train_pl = tf.placeholder(tf.float32, [None, 30])
test_pl = tf.placeholder(tf.float32, [30])

knn_prediction = tf.reduce_sum(tf.abs(tf.add(train_pl, tf.negative(test_pl))), axis=1)

pred = tf.argmin(knn_prediction, 0)

with tf.Session() as tf_session:
    missed = 0

    for i in range(len(test_dataset)):
        knn_index = tf_session.run(pred, feed_dict={train_pl: train_dataset, test_pl: test_dataset[i]})

        print("Predicted class {} -- True class {}".format(train_labels[knn_index], test_labels[i]))

        if train_labels[knn_index] != test_labels[i]:
            missed += 1

    tf.summary.FileWriter("D:/Tensorflow/Practice Code/KNN Dataset/samples/articles/logs", tf_session.graph)

print("Missed: {} -- Total: {}".format(missed, len(test_dataset)))