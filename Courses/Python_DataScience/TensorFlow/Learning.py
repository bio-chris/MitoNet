import tensorflow as tf

hi = tf.constant("Hello World")

sess = tf.Session()

print(sess.run(hi))

# Operations

x = tf.constant(2)
y = tf.constant(3)

with tf.Session() as sess:
    print("Operations with constants")
    print(sess.run(x+y))




