import tensorflow as tf
import numpy as np  #导入科学计算的模块

# create data
x_data = np.random.rand(100).astype(np.float32) #随机生成100个float32型的数字
y_data = x_data*0.1 + 0.3   #预测结果是weight权重要接近0.1,biases偏值要接近0.3

### create tensorflow structure start ###
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))    #定义权重,一维,初始值范围是-1.0～1.0
biases = tf.Variable(tf.zeros([1])) #定义biases,初始值为0

y = Weights*x_data + biases #预测的值,机器学习的目的就是要提升y的准确度

loss = tf.reduce_mean(tf.square(y-y_data))      #求平均值
optimizer = tf.train.GradientDescentOptimizer(0.5)  #实现梯度下降算法的优化器
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)

for step in range(201):   #range(201)代表从0到201,不包含201
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))