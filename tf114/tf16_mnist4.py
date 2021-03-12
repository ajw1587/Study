import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255

x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

# 2. 모델구성
# w1 = tf.Variable(tf.random_normal([784, 100]), name = 'weight1')
w1 = tf.get_variable('weight1', shape = [784, 100],
                     initializer = tf.contrib.layers.xavier_initializer())      # kernel_initializer
b1 = tf.Variable(tf.random_normal([100]), name = 'bias1')
print('w1 : ', w1)
print('b1 : ', b1)
# w1 :  <tf.Variable 'weight1:0' shape=(784, 100) dtype=float32_ref>
# b1 :  <tf.Variable 'bias1:0' shape=(100,) dtype=float32_ref>
layer1 = tf.nn.elu(tf.matmul(x, w1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob = 0.3)

w2 = tf.get_variable('weight2', shape = [100, 128],
                     initializer = tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([128]), name = 'bias2')
layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob = 0.3)

w3 = tf.get_variable('weight3', shape = [128, 64],
                     initializer = tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([64]), name = 'bias3')
layer3 = tf.nn.selu(tf.matmul(layer2, w3) + b3)
layer3 = tf.nn.dropout(layer3, keep_prob = 0.3)

w4 = tf.get_variable('weight4', shape = [64, 10],
                     initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([10]), name = 'bias4')
hypothesis = tf.nn.softmax(tf.matmul(layer3, w4) + b4)


# 컴파일, 훈련(다중분류)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(cost)

training_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)      # 60000 / 100 = 600

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epoch in range(training_epochs):
    avg_cost = 0

    for i in range(total_batch):
        start = i * batch_size
        end = start + batch_size

        batch_x, batch_y = x_train[start:end], y_train[start:end]
        feed_dict = {x:batch_x, y:batch_y}
        c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
        avg_cost += c/total_batch

    print('Epoch: ', '%04d' %(epoch + 1),
          'cost = {:.9f}'.format(avg_cost))
prediction = tf.equal(tf.arg_max(hypothesis, 1), tf.argmax(y, 1))
print('Prediction: ', sess.run(prediction, feed_dict = {x:x_test, y:y_test}))

accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
print('Acc: ', sess.run(accuracy, feed_dict={x:x_test, y:y_test}))

# Prediction:  [False False False ... False False False]
# Acc:  0.1179