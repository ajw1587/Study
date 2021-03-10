import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(77)

# predict
# 73, 80, 75, 152
# 93, 88, 93, 185
# 89, 91, 90, 180
# 96, 98, 100, 196
# 73, 66, 70, 142

# x_test, y_test 나눠주시
test_dataset = np.array([[73, 80, 75, 152],
                         [93, 88, 93, 185],
                         [89, 91, 90, 180],
                         [96, 98, 100, 196],
                         [73, 66, 70, 142]])
print(test_dataset.shape)
x_test = test_dataset[:, :-1]
y_test = test_dataset[:, -1:]


# 1. 데이터 로드
dataset = np.loadtxt('../data/csv/data-01-test-score.csv', delimiter = ',', dtype = 'float')
# print(type(dataset))                    # <class 'numpy.ndarray'>

# 2. 데이터 나누기
x_data = dataset[:, :-1]
y_data = dataset[:, -1:]
# print(x_data[0])                        # [73 80 75]
# print(y_data[0])                        # 152
# print(x_data.shape)                     # (25, 3)
# print(y_data.shape)                     # (25, 1)

# 3. 변수 만들기
x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# 4. Hypothesis
hypothesis = tf.matmul(x, w) + b

# 5. Cost Function
cost = tf.reduce_mean(tf.square(hypothesis - y))

# 6. Optimizer
train = tf.train.GradientDescentOptimizer(learning_rate = 0.00001).minimize(cost)
# train = tf.train.AdamOptimizer(learning_rate = 0.00001).minimize(cost)

# 7. train
epochs = 3000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    
    for epoch in range(epochs):
        _, Cost, Hypo = sess.run([train, cost, hypothesis],
                                 feed_dict = {x:x_data, y:y_data})
        if epoch % 2999 == 0:
            print('\nEpoch: ', epoch, '\nW: \n', sess.run(w), '\nb: ', sess.run(b), '\nhypo: \n', Hypo, '\ncost: \n', Cost)
    test_hypo = sess.run(hypothesis, feed_dict = {x:x_test})
    print('\nTEST_HYPO\n', test_hypo)
        

# Epoch:  2999 
# W:
#  [[0.32919627]
#  [0.6487573 ]
#  [1.0488846 ]]
# b:  [-0.95340586]
# hypo:
#  [[153.64517 ]
#  [184.29861 ]
#  [181.78174 ]
#  [199.11615 ]
#  [139.31769 ]
#  [104.025116]
#  [150.53326 ]
#  [113.782234]
#  [173.33769 ]
#  [162.76724 ]
#  [143.74268 ]
#  [141.87668 ]
#  [186.74257 ]
#  [153.52264 ]
#  [151.2625  ]
#  [188.09395 ]
#  [144.7052  ]
#  [181.64594 ]
#  [177.90598 ]
#  [159.33524 ]
#  [176.23338 ]
#  [173.90607 ]
#  [167.72598 ]
#  [152.38379 ]
#  [190.6279  ]]
# cost:
#  6.4255886