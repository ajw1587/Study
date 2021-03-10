import tensorflow as tf
tf.compat.v1.set_random_seed(77)

x_data = [[73, 51, 65],
          [92, 98, 11],
          [89, 31, 33],
          [99, 33, 100],
          [17, 66, 79]]
y_data = [[152],
          [185],
          [180],
          [205],
          [142]]

x = tf.placeholder(tf.float32, shape = [None, 3])
y = tf.placeholder(tf.float32, shape = [None, 1])

w = tf.Variable(tf.random_normal([3, 1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

# hypothesis = x*w + b          # 행렬 곱이라 에러가 뜬다.
hypothesis = tf.matmul(x, w) + b

# [실습] verbose로 나오는 놈이 step과 cost와 hypothesis를 출력
cost = tf.reduce_mean(tf.square(hypothesis - y))

# train = tf.train.GradientDescentOptimizer(learning_rate = 0.00001).minimize(cost)
train = tf.train.AdamOptimizer(learning_rate = 0.01).minimize(cost)

epochs = 3000
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for epoch in range(epochs):
        _, hypo_val, cost_val = sess.run([train, hypothesis, cost], feed_dict = {x:x_data, y:y_data})
        if epoch % 2999 == 0:
            print('Epoch: ', epoch, '\nW: \n', sess.run(w), '\nb: ', sess.run(b), '\nhypo: \n', hypo_val, '\ncost: \n', cost_val, '\n')

# GradientDescentOptimizer
# hypo:
#  [[171.25276]
#  [191.35498]
#  [151.62186]
#  [212.98128]
#  [127.0262 ]]

# AdamOptimizer
# hypo:
#  [[171.31424 ]
#  [191.10197 ]
#  [152.27568 ]
#  [212.51529 ]
#  [127.373276]]