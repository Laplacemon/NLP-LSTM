import pickle
from keras.layers.core import Activation, Dropout, Dense, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import jieba  # 用来分词
import numpy as np
import pandas as pd
import json

# 加载分词字典
with open('model/word_dict.pickle', 'rb') as handle:
    word2index = pickle.load(handle)

# 准备数据
MAX_FEATURES = 40002  # 最大词频数
MAX_SENTENCE_LENGTH = 80  # 句子最大长度
num_recs = 0  # 样本数

with open("data/train.json", "r", encoding="utf-8", errors='ignore') as f:
    lines = json.load(f)
    f.close()
    for line in lines:
        num_recs += 1

# 初始化句子数组和label数组
X = np.empty(num_recs, dtype=list)
y = np.zeros(num_recs)
i = 0

with open("data/train.json", "r", encoding="utf-8", errors='ignore') as f:
    lines = json.load(f)
    f.close()
    for line in lines:
        sentence = line[0].replace(' ', '')
        label = line[1]
        words = jieba.cut(sentence)
        seqs = []
        for word in words:
            # 在词频中
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])  # 不在词频内的补为UNK
        X[i] = seqs
        y[i] = int(label)
        i += 1

# 把句子转换成数字序列，并对句子进行统一长度，长的截断，短的补0
X = sequence.pad_sequences(X, maxlen=MAX_SENTENCE_LENGTH)
# 使用pandas对label进行one-hot编码
y1 = pd.get_dummies(y).values
print(X.shape)
print(y1.shape)
# 数据划分
x_train, x_test, y_train, y_test = train_test_split(X, y1, test_size=0.2, random_state=0, shuffle=True)

# 网络构建
EMBEDDING_SIZE = 40  # 词向量维度
HIDDEN_LAYER_SIZE = 20  # 隐藏层大小
MAX_FEATURES = 40001  # 最大词频数
BATCH_SIZE = 1024  # 每批大小
NUM_EPOCHS = 15  # 训练周期数

# 创建一个实例
model = Sequential()
# 构建词向量, 词向量长度：EMBEDDING_SIZE
model.add(Embedding(MAX_FEATURES, EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))
model.add(SpatialDropout1D(0.25))
# 构建LSTM层: 输入为（batches,samples,steps)
model.add(
    LSTM(
        HIDDEN_LAYER_SIZE,
        dropout=0.1,
        recurrent_dropout=0.1,
        activation='tanh',
    )
)
model.add(Dropout(0.25))
model.add(Dense(6, activation='relu'))
# 输出激活函数
model.add(Activation('softmax'))
# 损失函数设置为分类交叉熵 categorical_cross entropy
from keras import losses

model.compile(loss=losses.CategoricalCrossentropy(), optimizer="nadam",
              metrics=["accuracy"])
model.summary()

his = model.fit(
    x_train,
    y_train,
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    validation_split=0.1,
    validation_batch_size=BATCH_SIZE // 10,
)

import matplotlib.pyplot as plt

fig = plt.figure('Loss')
ax = fig.add_axes([0.1, 0.1, 0.8, 0.75])
for key in his.history.keys():
    if 'loss' in key:
        ax.plot(his.history[key], marker='o', label=key)
        ax.legend()

plt.title(f"Training result on 'LSTM_layer_size: {HIDDEN_LAYER_SIZE},\n batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}' ")
plt.xlabel('Epochs')
plt.ylabel('Categorical_cross Entropy')
plt.show()

fig = plt.figure('Acc')
ax = fig.add_axes([0.1, 0.1, 0.8, 0.75])
for key in his.history.keys():
    if 'loss' not in key:
        ax.plot(his.history[key], marker='o', label=key)
        ax.legend()

plt.title(f"Training result on 'LSTM_layer_size: {HIDDEN_LAYER_SIZE},\n batch_size={BATCH_SIZE}, epochs={NUM_EPOCHS}' ")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# 评估模型
y_predict = model.predict(x_test)
y_predict = y_predict.argmax(axis=1)
y_test = y_test.argmax(axis=1)

print('accuracy %s' % accuracy_score(y_predict, y_test))
target_names = ['其它', '喜好', '悲伤', '厌恶', '愤怒', '高兴']
print(classification_report(y_test, y_predict, target_names=target_names))

print("保存模型")
model.save('model/my_model.h5')
