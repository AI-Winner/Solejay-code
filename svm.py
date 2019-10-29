from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import pickle
import numpy as np
import csv

# 读取文件

# test 文件
with open("security_test.csv.pkl", "rb") as f:
    file_names = pickle.load(f) # list 12955
    outfiles = pickle.load(f) # list 12955
# train 文件
with open("security_train.csv.pkl", "rb") as f:
    labels = pickle.load(f) # ndarray (13887,)
    files = pickle.load(f) # list 13887

# 向量化表示
vectorizer = CountVectorizer(ngram_range=(1, 3))
x_train = vectorizer.fit_transform(files) # (13887, 180858) 
x_test = vectorizer.transform(outfiles)  # (12955, 180858)
y_train = labels # (13887,)

# SVM 分类
svc = SVC(gamma='auto', probability=True, decision_function_shape='ovo')
svc.fit(x_train, y_train)
result = svc.predict_proba(x_test)

# 结果整合
out = []
for i in range(len(file_names)):
    print(i)
    tmp = []
    a = result[i].tolist() # ndarray 转化为 list
    # for j in range(len(a)):
    #     a[j] = ("%.5f" % a[j])

    tmp.append(file_names[i])
    tmp.extend(a)
    out.append(tmp)

# 输出csv文档
with open("svm.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
                     ])
    # 写入多行用writerows
    writer.writerows(out)