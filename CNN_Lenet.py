#######################################最简单的Lenet，使用mnist手写子体作为训练集。
###      https://zhuanlan.zhihu.com/p/26648813


import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.models import Sequential

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape) # x是训练集
print(x_test.shape)
print(y_train.shape) # y是标签列
print(y_test.shape)

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train/255.0
x_test = x_test/255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)


lenet = Sequential()
lenet.add(Conv2D(filters=6, kernel_size=3, strides=1, padding='same', input_shape=(28, 28, 1)))
lenet.add(MaxPool2D(pool_size=2, strides=2))
lenet.add(Conv2D(filters=16, kernel_size=5, strides=1, padding='valid'))
lenet.add(MaxPool2D(pool_size=2, strides=2))
lenet.add(Flatten())
lenet.add(Dense(120))
lenet.add(Dense(84))
lenet.add(Dense(10, activation='softmax'))


lenet.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])
lenet.fit(x_train, y_train, batch_size=64, epochs=50, validation_data=[x_test, y_test])
