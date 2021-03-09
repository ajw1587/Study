# tensorflow 1버전
# constant, placeholder, value
import tensorflow as tf
print(tf.__version__)

hello = tf.constant('Hello World')
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

sess = tf.Session()
print(sess.run(hello))
# Session을 통과 시켜야 제대로 나온다.
# 통과시키지 않을경우 변수의 정보만 출력된다.