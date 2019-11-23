# 读取训练集和测试集
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import pickle
import numpy as np
import time
import csv

# train 文件
with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f) # ndarray (13887,)
    train_apis = pickle.load(f)  # list 13887

# test 文件
with open("security_test.csv.pkl", "rb") as f:
    test_nums = pickle.load(f) # list 12955
    test_apis = pickle.load(f) # list 12955

# 词袋模型 ngram 提取特征
vectorizer = CountVectorizer(ngram_range=(1, 3))
x_train = vectorizer.fit_transform(train_apis)  # (13887, 180858) 
y_train = labels # (13887,)
x_test = vectorizer.transform(test_apis)  # (12955, 180858)

# 将词向量保存下来
with open("ngram_model/ngram(1,3)_vec.pkl", 'wb') as f:
    pickle.dump(x_train, f)
    pickle.dump(y_train, f)
    pickle.dump(x_test, f)
    pickle.dump(test_nums, f)

# 加载词向量
with open("ngram_model/ngram(1,3)_vec.pkl", 'rb') as f:
    x_train = pickle.load(f) # (13887, 180858) 
    y_train = pickle.load(f) # (13887,)
    x_test = pickle.load(f) # (12955, 180858)
    test_nums = pickle.load(f) # list 12955

# xgboost对词向量模型进行分类
ngram_train = np.zeros(shape=(13887, 8))  # 训练集（验证集）预测结果
ngram_test = np.zeros(shape=(12955, 8)) # 测试集预测结果

k = 10 # 交叉验证的折数
skf = StratifiedKFold(n_splits=k, random_state=42, shuffle=True)
for i, (tr_ind, te_ind) in enumerate(skf.split(x_train, y_train)):
    X_train, X_train_label = x_train[tr_ind], y_train[tr_ind]
    X_val, X_val_label = x_train[te_ind], y_train[te_ind]

    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind)) # 测试集和训练集数量

    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    dtest = xgb.DMatrix(X_val, X_val_label)
    dout = xgb.DMatrix(x_test)

    param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
             'num_class': 8, 'subsample': 1,
             'colsample_bytree': 0.85}  # 参数

    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 300  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    # dtr = xgb.DMatrix(train_features)
    pred_val = bst.predict(dtest)
    pred_test = bst.predict(dout)
    ngram_train[te_ind] = pred_val
    ngram_test += pred_test
ngram_test /= k * 1.0

# 保存训练集和测试集预测结果
with open("ngram_model/ngram_result_seed42.pkl", 'wb') as f:
    pickle.dump(ngram_train, f)
    pickle.dump(ngram_test, f)

# 输出 csv 文件
result = ngram_test
out = []

for i in range(len(test_nums)):
    tmp = []
    a = result[i].tolist()

    tmp.append(test_nums[i])
    tmp.extend(a)
    out.append(tmp)
    
with open("result/xgboost(1,3)_10折_seed42.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
                     ])
    # 写入多行用writerows
    writer.writerows(out)