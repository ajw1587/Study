# tensorflow 1점대 실행 (가상환경 tf114)
import tensorflow as tf

sess = tf.Session()

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print('sess.run(node1, node2): ', sess.run([node1, node2]))
print('node3: ', node3)
print('sess.run(node3): ', sess.run(node3))
# sess.run(node1, node2):  [3.0, 4.0]
# node3:  Tensor("Add:0", shape=(), dtype=float32)
# sess.run(node3):  7.0

# 텐서플로우 기본개념: https://blog.naver.com/complusblog/221237818389