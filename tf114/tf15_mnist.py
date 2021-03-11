from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

unique, counts = np.unique(y_train, return_counts = True)
# print(np.array((unique, counts)).T)           # 0~9 총 10개

x_train = x_train.reshape(60000, 784).astype('float32')/255
x_test = x_test.reshape(10000, 784).astype('float32')/255
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# OneHotEncoder 사용하면 에러 발생
enc = OneHotEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train)
y_test = enc.transform(y_test)
y_train = y_train.toarray()
y_test = y_test.toarray()
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)

print(x_train.shape)                          # (60000, 784)
print(y_train.shape)                          # (60000, 10)
print(x_test.shape)                           # (10000, 784)
print(y_test.shape)                           # (10000, 10)
print(type(y_train))                          # <class 'scipy.sparse.csr.csr_matrix'>
print(type(y_test))                           # <class 'scipy.sparse.csr.csr_matrix'>


# x, y 선언
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float', [None, 10])

w1 = tf.Variable(tf.random_normal([784, 64]), name = 'weight1')
b1 = tf.Variable(tf.random_normal([64]), name = 'bias1')
layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([64, 64]), name = 'weight2')
b2 = tf.Variable(tf.random_normal([64]), name = 'bias2')
layer2 = tf.matmul(layer1, w2) + b2

w3 = tf.Variable(tf.random_normal([64, 64]), name = 'weight3')
b3 = tf.Variable(tf.random_normal([64]), name = 'bias3')
layer3 = tf.matmul(layer2, w3) + b3

w4 = tf.Variable(tf.random_normal([64, 32]), name = 'weight4')
b4 = tf.Variable(tf.random_normal([32]), name = 'bias4')
layer4 = tf.matmul(layer3, w4) + b4

w5 = tf.Variable(tf.random_normal([32, 10]), name = 'weight5')
b5 = tf.Variable(tf.random_normal([10]), name = 'bias5')
hypothesis = tf.nn.softmax(tf.matmul(layer4, w5) + b5)

# cost, optimizer
cost =  tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))
train = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)


# train
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(5001):
        _, cost_val = sess.run([train, cost], feed_dict = {x:x_train, y:y_train})
        if epoch % 200 == 0:
            print(epoch, cost_val)
    
    a = sess.run(hypothesis, feed_dict = {x:x_test})
    print(a, sess.run(tf.argmax(a, 1)))
    