import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Data
x_data = load_wine().data
y_data = load_wine().target

# sns.countplot(y_data)
# plt.show()            # y_data = 0, 1, 2

# OneHotEncoder
enc = OneHotEncoder()
y_data = y_data.reshape(-1, 1)
enc.fit(y_data)
y_data = enc.transform(y_data).toarray()

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size = 0.8,
                                                    random_state = 77)
# print(x_train.shape, y_train.shape)           # (142, 13) (142, 3)
# print(x_test.shape, y_test.shape)             # (36, 13) (36, 3)

# MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# x, y, w, b
x = tf.placeholder('float', shape = [None, 13])
y = tf.placeholder('float', shape = [None, 3])

w = tf.Variable(tf.random_normal([13, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')

# hypothesis, cost
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

# train = tf.train.GradientDescentOptimizer(learning_rate = 0.0001).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

# train
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(2001):
        _, cost_val = sess.run([train, cost], feed_dict = {x:x_train, y:y_train})

        if epoch % 200 == 0:
            print(epoch, '\t', cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x:x_test, y:y_test})
    print('예측값:\n', h, '\n원래값:\n', c, '\nAccuracy: ', a)

    y_predict = sess.run(hypothesis, feed_dict={x:x_test})
    print("accuracy_score : ", accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))

# Accuracy:  1.0
# accuracy_score :  1.0