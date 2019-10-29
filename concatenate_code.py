'''
特征提取和模型使用整合到同一个文件中，各自完整代码参照其他文件
采用ngram分词或TFIDF提取特征，范围为（1,3）
StratifiedKFold 5折交叉验证
xgboost进行分类

svm进行分类
'''
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import pickle
import numpy as np
import time
import csv

# test 文件
with open("../test/security_test.csv.pkl", "rb") as f:
    file_names = pickle.load(f)
    outfiles = pickle.load(f)  

# train 文件
with open("../test/security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f)
    files = pickle.load(f)  

# 词袋模型 ngram 提取特征
vectorizer = CountVectorizer(ngram_range=(1, 3))
x_train = vectorizer.fit_transform(files)  # (13887, 180858) 
y_train = labels # (13887,)
x_test = vectorizer.transform(outfiles)  # (12955, 180858)

# # Tfidf 提取特征
# vectorizer = TfidfVectorizer(ngram_range=(1,3), min_df=3, max_df=0.9, )  
# train_features = vectorizer.fit_transform(files)  # (13887, 180858) 
# y_train = labels # (13887,)
# out_features = vectorizer.transform(outfiles)  # (12955, 180858)



# xgboost 模型
meta_train = np.zeros(shape=(len(files), 8))  
meta_test = np.zeros(shape=(len(outfiles), 8))  
# StratifiedKFold用法类似 Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
skf = StratifiedKFold(n_splits=5, random_state=4, shuffle=True)
for i, (tr_ind, te_ind) in enumerate(skf.split(x_train, labels)):
    X_train, X_train_label = x_train[tr_ind], labels[tr_ind]
    X_val, X_val_label = x_train[te_ind], labels[te_ind]

    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind)) # 测试集和训练集数量

    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    dtest = xgb.DMatrix(X_val, X_val_label)
    dout = xgb.DMatrix(x_test)

    param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
             'num_class': 8, 'subsample': 0.8,
             'colsample_bytree': 0.85}  # 参数

    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 300  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    # dtr = xgb.DMatrix(train_features)
    pred_val = bst.predict(dtest)
    pred_test = bst.predict(dout)
    meta_train[te_ind] = pred_val
    meta_test += pred_test
meta_test /= 5.0
result = meta_test

# SVM 模型
svc = SVC(gamma='auto', probability=True, decision_function_shape='ovo')
svc.fit(x_train, y_train)
result = svc.predict_proba(x_test)

out = []
for i in range(len(file_names)):
    tmp = []
    a = result[i].tolist()
    # for j in range(len(a)):
    #     a[j] = ("%.5f" % a[j])

    tmp.append(file_names[i])
    tmp.extend(a)
    out.append(tmp)
with open("../test/ngram_3gram.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
                     ])
    # 写入多行用writerows
    writer.writerows(out)