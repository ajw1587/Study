from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import tensorflow as tf

# [실습] r2_score 값으로 결론
# Data
dataset = load_diabetes()

x_train, x_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, train_size = 0.8, random_state = 77
)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
print(x_train.shape)        # (353, 10)
print(x_test.shape)         # (89, 10)
print(y_train.shape)        # (353, 1)
print(y_test.shape)         # (89, 1)

# Model
x = tf.placeholder(tf.float32, shape = [None, 10])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([10, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# Hypothesis
hypothesis = tf.matmul(x, w) + b

# Cost
cost = tf.reduce_mean(tf.square(hypothesis - y))

# Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(cost)

# Train
epochs = 10001
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for epoch in range(epochs):
        _, hypo, Cost = sess.run([train, hypothesis, cost],
                             feed_dict = {x:x_train, y:y_train})
        if epoch % 500 == 0:
            print('EPOCH: ', epoch, '\tCost: ', Cost)
    
    y_predict = sess.run(hypothesis, feed_dict = {x:x_test})
    print('y_predict: \n', y_predict)