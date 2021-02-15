from sklearn.datasets import load_iris
from sklearn import datasets
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('expand_frame_repr', False)

###导入sklearn默认数据集
# x_data = datasets.load_iris().data
# y_data = datasets.load_iris().target


####################################导入本地数据##############################################
"""
关键是将本地数据集转化成sklearn默认数据集的格式
"""
df = pd.read_csv(r'C:\Users\47053\Desktop\data_iris.csv', sep=',')

data = df.values

x_data = [lines[0:4] for lines in data]
y_data = [lines[4] for lines in data]

x_data = np.array(x_data, dtype=float)

for i in range(len(x_data)):
    if y_data[i] == 'setosa':
        y_data[i] = 0
    elif y_data[i] == 'versicolor':
        y_data[i] = 1
    else:
        y_data[i] = 2

y_data = np.array(y_data)

#############################################################################################


np.random.seed(116)
np.random.shuffle(x_data)

np.random.seed(116)
np.random.shuffle(y_data)

np.random.seed(116)
x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

w1 = tf.Variable(tf.random.truncated_normal([4, 3], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3], stddev=0.1, seed=1))

lr = 0.1
train_loss_res = []
test_acc = []
epoch = 500
loss_all = 0

# 训练
for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)
            y_pred = tf.one_hot(y_train, depth=3)
            loss = tf.reduce_mean(tf.square(y - y_pred))
            loss_all += loss.numpy()

        grads = tape.gradient(loss, [w1, b1])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])

    print("Epoch {}, loss: {}".format(epoch+1, loss_all / 4))
    train_loss_res.append(loss_all/4)
    loss_all = 0

    # 测试
    total_correct, total_number = 0, 0
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        # y = tf.squeeze(y)  # 添加了这一行
        y = tf.nn.softmax(y)
        y_pred = tf.argmax(y, axis=1)
        y_pred = tf.cast(y_pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(y_test, y_pred), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]  # batch1 32, batch2 32, batch3 32, batch4 26
    acc = total_correct / total_number

    test_acc.append(acc)
    print('Test acc: ', acc)
    print('--------------------------------')


plt.title("Loss function Curve")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_res, label="$Loss$")
plt.legend()
plt.show()


plt.title("Acc Curve")
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label="$Accuracy$")
plt.legend()
plt.show()
