# -*-coding:utf-8-*-
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pickle


# 加载真实结果
with open("../security_train.csv.pkl", "rb") as f:
    y_true = pickle.load(f) # ndarray (13887,)
    train_apis = pickle.load(f)  # list 13887


# 加载预测结果
with open("../ngram_result_seed42.pkl", 'rb') as f:
    ngram_train = pickle.load(f)
    ngram_test = pickle.load(f)


#labels表示你不同类别的代号，比如这里的demo中有13个类别
labels = [0, 1, 2, 3, 4, 5, 6, 7]
y_true = y_true.astype(int)  # string 转换为 int 类型
y_pred = ngram_train.argmax(axis=1)



tick_marks = np.array(range(len(labels))) + 0.5


def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')
# show confusion matrix
plt.savefig('confusion_matrix.png', format='png')
plt.show()


