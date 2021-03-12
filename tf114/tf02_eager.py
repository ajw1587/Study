# tensorflow 1버전
# 즉시 실행 모드: Session 정의 후 run 유무
import tensorflow as tf
# from tensorflow.python.framework.ops import disable_eager_execution

# tensorflow 2점대에서 실행
print(tf.executing_eagerly())               # True
tf.compat.v1.disable_eager_execution()      # 즉시 실행 모드를 OFF 한다.
print(tf.executing_eagerly())               # False

print(tf.__version__)


hello = tf.constant('Hello World')
print(hello)
# Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()                      # 1.13v 까지의 Session
sess = tf.compat.v1.Session()              # 1.14v 의 Session
print(sess.run(hello))
# Session을 통과 시켜야 제대로 나온다.
# 통과시키지 않을경우 변수의 정보만 출력된다.
# 2점대의 tensorflow에서는 Session 자체가 없다.