# Solejay-code
Renjie Pu's code

### load_file.py

读取训练集和测试集数据，处理后保存每个文件对应的 `label` 和 `拼接的 api 序列` 并保存为 `pkl` 文件实现数据的持久存储。

### train.py

##### 特征提取

- ngram 
- TFIDF

##### 模型选择

- xgboost 模型
- svm 模型

##### 模型优化

xgboost 进行 5 折交叉验证选取效果最好的模型

```python
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
```

svm 模型

```python
svc = SVC(gamma='auto', probability=True, decision_function_shape='ovo')
svc.fit(x_train, y_train)
result = svc.predict_proba(x_test)
```

### 提交结果

- 词袋模型+xgboost 5折交叉验证（未填写的参数为默认参数）

unigram：0.567571（ngram_range=(1, 1)）

3_gram：0.473048（ngram_range=(1, 3)）

4_gram：0.473261（ngram_range=(1, 4)）

5_gram：0.473122（ngram_range=(1, 5)）

(1,3)_gram：0.476422（ngram_range=(1, 3), min_df=3, max_df=0.9）

> train-mlogloss:0.070641 val-mlogloss:0.2973

(2, 3)_gram：0.484424（ngram_range=(2, 3)）

>  train-mlogloss:0.074861 val-mlogloss:0.297651

(2, 4)_gram：0.482929（ngram_range=(2, 4)）

> train-mlogloss:0.077568 val-mlogloss:0.294211

- TFIDF模型+xgboost 5折交叉验证

unigram：0.659438（ngram_range=(1, 1)）

3_gram：0.505826（ngram_range=(1, 3)）

5_gram：0.507149（ngram_range=(1, 5)）

### 结果总结

1. ngram 模型效果优于 IFIDF 模型
2. ngram_range（1, 3）效果优于（1, 1）和（1, 5）
3. ngram_range 从 1 开始 优于 从 2 开始
4. min_df 和 max_df 参数修改后效果变差