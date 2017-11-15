import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output1 = tf.multiply(input1, input2)

with tf.Session() as sess:
    result = sess.run(output1,feed_dict={input1:[5.], input2: [6.]})
    print(result)