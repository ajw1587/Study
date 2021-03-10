# 이진분류
# [실습] accuray_score 값으로 결론
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import tensorflow as tf

# 1. Data
dataset = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state = 77)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# print(x_train.shape)        # (455, 30)
# print(y_train.shape)        # (455, 1)
# print(x_test.shape)         # (114, 30)
# print(y_test.shape)         # (114, 1)

# 2. Weight, Bias
x = tf.placeholder(tf.float32, shape = [None, 30])
y = tf.placeholder(tf.float32, shape = [None, 1])
w = tf.Variable(tf.random_normal([30, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 3. Hypothesis, Cost, Optimizer
hypothesis = tf.sigmoid(tf.matmul(x, w) + b)
cost = -tf.reduce_mean(y*tf.log(hypothesis) + (1-y)*tf.log(1-hypothesis))

train = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

# 4. Train
epochs = 3001
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(epochs):
        _, cost_val = sess.run([train, cost], feed_dict = {x:x_train, y:y_train})
        if epoch % 500 == 0:
            print(epoch, cost_val)

    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict = {x:x_test, y:y_test})
    print('예측값:\n', h, '\n원래값:\n', c, '\nAccuracy: ', a)

    y_predict = sess.run(hypothesis, feed_dict={x:x_test})
    print("accuracy_score : ", accuracy_score(y_test, sess.run(predicted, feed_dict={x:x_test})))