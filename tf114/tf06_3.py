# [실습] tf06_2.py의 lr을 수정해서
# epoch가 2000번보다 적게 만들어라

import tensorflow as tf
tf.set_random_seed(77)

# x_train = [1, 2, 3]
# y_train = [3, 5, 7]

x_train = tf.placeholder(tf.float32, shape = [None])
y_train = tf.placeholder(tf.float32, shape = [None])

# tf.random_normal: 정규분포를 만족하는 데이터 생성
W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(W), sess.run(b))

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
# train = optimizer.minimize(cost)
train = tf.train.GradientDescentOptimizer(learning_rate = 0.175).minimize(cost)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# for step in range(2001):
#     sess.run(train)
#     if step % 20 == 0:
#         print(step, sess.run(cost), sess.run(W), sess.run(b))
# # 초기 W: 1.1317381
# # 초기 b: 0.41270408
# sess.close()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b], 
                                             feed_dict = {x_train:[1, 2, 3], y_train:[3, 5, 7]})
        if step % 100 == 0:
            print(step, cost_val, W_val, b_val)