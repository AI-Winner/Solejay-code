import pandas as pd
import csv

src1 = 'result/xgboost(1,3)_seed0.csv'
src2 = 'result/xgboost(1,3)_seed8.csv'
src3 = 'result/xgboost(1,3)_seed42.csv'
src4 = 'result/xgboost(1,3).csv'
out = []

res1 = pd.read_csv(src1)
res2 = pd.read_csv(src2)
res3 = pd.read_csv(src3)
res4 = pd.read_csv(src4)

# 去掉 file_id 列相加求和除以数量求和
res = (res1.iloc[:, 1:] + res2.iloc[:, 1:] + res3.iloc[:, 1:] + res4.iloc[:, 1:]) / 4.0

for i in range(res.shape[0]):
    print(i)
    tmp = []
    a = pd.Series.tolist(res.iloc[i]) # Series转换为list
    tmp.append(i+1) # 添加每行对应的 file_id
    tmp.extend(a)
    out.append(tmp)

# 输出csv文档
with open("result/4_mean_seed.csv", "w", newline='') as csvfile:
    writer = csv.writer(csvfile)

    # 先写入columns_name
    writer.writerow(["file_id", "prob0", "prob1", "prob2", "prob3", "prob4", "prob5", "prob6", "prob7"
                    ])
    # 写入多行用writerows
    writer.writerows(out)






        

