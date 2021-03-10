import tensorflow as tf

x = [1, 2, 3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

# [실습]
# 1. sess.run()
# 2. InteractiveSession()
# 3. eval(session = sess)
# hypothesis를 출력

# 1. sess.run()
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
hypo = sess.run(hypothesis)
print('1. hypothesis: ', hypo)
sess.close()

# 2. Interactivesession()
sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
hypo = hypothesis.eval()
print('2. hypothesis: ', hypo)
sess.close()

# 3. eval(session = sess)
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
hypo = hypothesis.eval(session = sess)
print('3. hypothesis: ', hypo)
sess.close()