# 混淆矩阵代码
true_train = []
pred_train = []
meta_test = np.zeros(shape=(len(outfiles), 8))  
# StratifiedKFold用法类似 Kfold，但是他是分层采样，确保训练集，测试集中各类别样本的比例与原始数据集中相同
skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
for i, (tr_ind, te_ind) in enumerate(skf.split(x_train, labels)):
    X_train, X_train_label = x_train[tr_ind], labels[tr_ind]
    X_val, X_val_label = x_train[te_ind], labels[te_ind]

    
    meta_train = np.zeros(shape=(len(tr_ind), 8))
    meta_val = np.zeros(shape=(len(te_ind), 8))

    print('FOLD: {}'.format(str(i)))
    print(len(te_ind), len(tr_ind)) # 测试集和验证集数量

    dtrain = xgb.DMatrix(X_train, label=X_train_label)
    dtest = xgb.DMatrix(X_val, X_val_label)
    dout = xgb.DMatrix(x_test)

    param = {'max_depth': 6, 'eta': 0.1, 'eval_metric': 'mlogloss', 'silent': 1, 'objective': 'multi:softprob',
             'num_class': 8, 'subsample': 1,
             'colsample_bytree': 0.85}  # 参数

    evallist = [(dtrain, 'train'), (dtest, 'val')]  # 测试 , (dtrain, 'train')
    num_round = 10  # 循环次数
    bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=50)

    # dtr = xgb.DMatrix(train_features)
    pred_val = bst.predict(dtest)

    # 验证集真实值和预测值
    true_val = X_val_label
    pred_val = pred_val.argmax(axis=1)

    true_train = np.append(true_train, true_val) # 验证集真实标签求和
    print(true_train.shape)
    pred_train = np.append(pred_train, pred_val) # 验证集预测标签求和
    print(pred_train.shape)

    print(confusion_matrix(pred_val, true_val))

print(confusion_matrix(pred_train, true_train))