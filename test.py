import pickle
import numpy as np
import pandas as pd

#
# # 类别 4 数据
# with open("train_label_four.csv.pkl", "rb") as f:
#     labels_4 = pickle.load(f) # ndarray (53,)
#     train_apis_4 = pickle.load(f)  # list 53
#
# # 训练集全体
# with open("../security_train.csv.pkl", "rb") as f:
#     labels = pickle.load(f) # ndarray (13887,)
#     train_apis = pickle.load(f)  # list 13887
#
# labels_concate = np.append(labels, labels_4)
# train_apis_concate = np.append(train_apis, train_apis_4)

# 加入第 4 类数据测试
# label1 = np.array([1, 2, 3, 4, 5])
# label2 = np.array([6, 7, 8, 9, 10])
# labels = np.concatenate((label1, label2), axis=0)
#
# api1 = [['api1', 'api2', 'api3'], ['api4', 'api5', 'api6']]
# api2 = [['api7', 'api8', 'api9'], ['api10', 'api11', 'api12']]
# # apis = np.concatenate((api1, api2), axis=0)
# apis = np.vstack((api1, api2))

# word2vec 数据预处理
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
model = Word2Vec.load('word2vec.model')
vec = model.wv['NtQueryKey']
print(vec)



