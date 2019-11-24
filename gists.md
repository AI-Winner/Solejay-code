- 测试集类别预测数量统计

```python
with open("../ngram_result_seed42.pkl", 'rb') as f:
    ngram_train = pickle.load(f)
    ngram_test = pickle.load(f)

test_pred = ngram_test.argmax(axis=1)
pred_counts = pd.Series(test_pred).value_counts()
true_counts = [4978, 409, 643, 670, 122, 4288, 629, 1216]
```

- 混淆矩阵

```python
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
plt.figure(figsize=(12, 8), dpi=200)

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
plt.savefig('train_confuse_matrix.png', format='png')
plt.show()
```

- word2vec

```python
with open("train_label_four.csv.pkl", "rb") as f:
    labels_4 = pickle.load(f) # ndarray (53,)
    train_apis_4 = pickle.load(f)  # list 53


with open('word2vec_test.txt', 'w') as f:
    for sentence in train_apis_4:
        f.write(sentence)
        f.write('\n')


from gensim.models import Word2Vec

word2vec = Word2Vec(corpus_file='word2vec_test.txt')
word2vec.save('word2vec.model')
```

- 数据合并

```python
# 类别 4 数据
with open("train_label_four.csv.pkl", "rb") as f:
    labels_4 = pickle.load(f) # ndarray (53,)
    train_apis_4 = pickle.load(f)  # list 53

# 训练集全体
with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f) # ndarray (13887,)
    train_apis = pickle.load(f)  # list 13887

labels_concate = np.concatenate((labels, labels_4), axis=0)
train_apis_concate = np.concatenate((np.array(train_apis), np.array(train_apis_4)), axis=0)
```

