from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import xgboost_model as xgb
import pickle
import numpy as np
import time
import csv

# test 文件
with open("security_test.csv.pkl", "rb") as f:
    file_names = pickle.load(f)
    outfiles = pickle.load(f)  

# train 文件
with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)  

# 词袋模型 ngram 提取特征
vectorizer = CountVectorizer(ngram_range=(1, 3))
x = vectorizer.fit_transform(files)  # (13887, 180858)
y = labels # (13887,)
x_test = vectorizer.transform(outfiles)  # (12955, 180858)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, y_test)
dout = xgb.DMatrix(x_test)

param = {'eval_metric': 'mlogloss', 'objective': 'multi:softprob',
         'num_class': 8}  # 参数

evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
num_round = 300  # 循环次数
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

# dtr = xgb.DMatrix(train_features)
pred_val = bst.predict(dtest)
pred_test = bst.predict(dout)
result = pred_test

out = []
for i in range(len(file_names)):
    tmp = []
    a = result[i].tolist()
    # for j in range(len(a)):
    #     a[j] = ("%.5f" % a[j])

    tmp.append(file_names[i])
    tmp.extend(a)
    out.append(tmp)
with open("xgboost_original_para.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
                     ])
    # 写入多行用writerows
    writer.writerows(out)