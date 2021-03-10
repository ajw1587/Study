import tensorflow as tf
# tf.set_random_seed(777)
tf.compat.v1.set_random_seed(777)

# W = tf.Variable(tf.random_normal([1]), name = 'weight')
W = tf.Variable(tf.compat.v1.random_normal([1]), name = 'weight')

print(W)
# <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>


# 1. tf.Session()
sess = tf.compat.v1.Session()                      # warning 메시지 사라짐
# sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())
aaa = sess.run(W)
print('aaa: ', aaa)
# [2.2086694]
sess.close()

# 2. tf.InteractiveSession()
sess = tf.compat.v1.InteractiveSession()           # warning 메시지 사라짐
# sess = tf.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())
bbb = W.eval()
print('bbb: ', bbb)
sess.close()

# 3. 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
ccc = W.eval(session = sess)
print('ccc: ', ccc)
sess.close()

# aaa:  [2.2086694]
# bbb:  [2.2086694]
# ccc:  [2.2086694]