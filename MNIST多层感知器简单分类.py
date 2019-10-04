####################################################日月光华(网易云、Blibli)###############################################################
import keras
from keras import layers
import matplotlib.pyplot as plt
import keras.datasets.mnist as mnist
import pylab

# print(mnist)

(train_image, train_label), (test_image, test_label) = mnist.load_data()

# print(train_image.shape)

# print(train_label[1000])
# plt.imshow(train_image[1000])
# plt.show()

# print(train_label)
# print(train_label.shape)

# print(test_image.shape, test_label.shape)



### 建立网络框架
model = keras.Sequential()
model.add(layers.Flatten()) # 添加第一层, 把(60000, 28, 28)-->(60000, 28*28)
model.add(layers.Dense(64, activation='relu'))  #添加第二层 输出64个单元的隐藏层
model.add(layers.Dense(64, activation='relu'))  #添加第三层 输出64个单元的隐藏层
model.add(layers.Dense(64, activation='relu'))  #添加第四层 输出64个单元的隐藏层
model.add(layers.Dense(10, activation='softmax')) # 添加第五层(分类层-输出层), 因为手写数字0-9 有10个，所以输出神经元个数为10, activation对应使用多酚类函数softmax


### 编译网络
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['acc'],
             )


### 训练模型                                                                     # 训练模型的时候，也可以看到测试模型结果
model.fit(train_image, train_label, epochs=50, batch_size=512, validation_data=(test_image, test_label))  # 每个批次是用512张图片来训练，把所有图片训练50次，


### 测试模型
# print(model.evaluate(test_image, test_label)) # eval = model.evaluate() 会返回两个值,eval[0]是loss, eval[1]是accuracy.
# print(model.evaluate(train_image, train_label))



# ### 用模型做预测
import numpy as np
# print(model.predict(test_image[:10]))
print(np.argmax(model.predict(test_image[:10]), axis=1)) # 预测的前10的概率
print(test_label[:10])  # 实际前10的label
