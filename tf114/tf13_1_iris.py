import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# 1. Data
x_data = load_iris().data
y_data = load_iris().target
# print(x.shape)      # (150, 4)
# print(y.shape)      # (150,)

# OneHotEncoding
y_data = y_data.reshape(-1, 1)
enc = OneHotEncoder()
enc.fit(y_data)
y_data = enc.transform(y_data).toarray()

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data,
                                                    train_size = 0.8,
                                                    random_state = 77)
# print(x_train.shape, y_train.shape)         # (120, 4) (120, 3)
# print(x_test.shape, y_test.shape)           # (30, 4) (30, 3)

# sns.countplot(y_data)
# plt.show()

# 2. W, b
x = tf.placeholder('float', shape = [None, 4])
y = tf.placeholder('float', shape = [None, 3])

w = tf.Variable(tf.random_normal([4, 3]), name = 'weight')
b = tf.Variable(tf.random_normal([1, 3]), name = 'bias')

# 3. Model
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(hypothesis), axis = 1))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype = tf.float32))

# 4. train
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

# Accuracy:  0.9777778
# accuracy_score :  0.9666666666666667