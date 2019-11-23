'''
实现读取 security_train.csv 和 security_test.csv 文件
将文件按照 file_id 分组后
按照 tid 和 index 排序
分别用 labels 存放 file_id 对应的程序类别
和    train_apis  存放 file_id 包含的 api 序列
最后导出 pkl 文件
'''
import pandas as pd
import pickle
import numpy as np

train_path = r'security_train.csv'
test_path = r'security_test.csv'


def read_train_file(path):
    labels = [] # 训练集样本对应的类别集合
    train_apis = [] # 训练集样本拼接 api 序列
    data = pd.read_csv(path)  
    group_fileid = data.groupby('file_id')
    for file_name, file_group in group_fileid:
        print(file_name)  # 一共 13887 个文件
        file_labels = file_group['label'].values[0]  # 属于哪一类病毒
        result = file_group.sort_values(['tid', 'index'], ascending=True)  # 按照 tid 和 index 升序排列
        api_sequence = ' '.join(result['api'])  # 把 api 拼接起来
        labels.append(file_labels)
        train_apis.append(api_sequence)
    labels = np.asarray(labels)  # 转换为 numpy 数组
    with open("security_train.csv.pkl", 'wb') as f:
        pickle.dump(labels, f)
        pickle.dump(train_apis, f)


def read_test_file(path):
    test_nums = [] # 测试集样本数（从 1 到 12955）
    test_apis = [] # 测试集样本拼接 api 序列
    data = pd.read_csv(path)
    group_fileid = data.groupby('file_id')
    for file_name, file_group in group_fileid:
        print(file_name)  # 一共 12955 个文件
        result = file_group.sort_values(['tid', 'index'], ascending=True)
        api_sequence = ' '.join(result['api'])
        test_nums.append(file_name)
        test_apis.append(api_sequence)
    with open("security_test.csv.pkl", 'wb') as f:
        pickle.dump(test_nums, f)
        pickle.dump(test_apis, f)

print("read train file.....")
read_train_file(train_path)
print("read test file......")
read_test_file(test_path)

