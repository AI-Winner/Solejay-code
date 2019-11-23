# Solejay-code
Renjie Pu's code

### load_file.py

读取训练集和测试集数据，处理后保存每个文件对应的 `label` 和 `拼接的 api 序列` 并保存为 `pkl` 文件实现数据的持久存储。

### xgboost.py

ngram + xgboost 分类器代码

### svm.py

ngram + svm 分类器代码

### mean.py

四个不同随机种子对模型进行训练的结果文件取平均

### 提交结果

- 词袋模型+xgboost 5折交叉验证（未填写的参数为默认参数）

unigram：**0.567571**（ngram_range=(1, 1)）

3_gram：**0.473048**（ngram_range=(1, 3)）

4_gram：**0.473261**（ngram_range=(1, 4)）

5_gram：**0.473122**（ngram_range=(1, 5)）

(1,3)_gram：**0.476422**（ngram_range=(1, 3), min_df=3, max_df=0.9）

> train-mlogloss:0.070641 val-mlogloss:0.2973

(2, 3)_gram：**0.484424**（ngram_range=(2, 3)）

>  train-mlogloss:0.074861 val-mlogloss:0.297651

(2, 4)_gram：**0.482929**（ngram_range=(2, 4)）

> train-mlogloss:0.077568 val-mlogloss:0.294211

- TFIDF模型+xgboost 5折交叉验证

unigram：**0.659438**（ngram_range=(1, 1)）

3_gram：**0.505826**（ngram_range=(1, 3)）

5_gram：**0.507149**（ngram_range=(1, 5)）

- ngram(ngram_range(1, 3)) + xgboost 5折交叉验证 + 随机种子

random_state=4：**0.473048**

> train-mlogloss:0.06956 val-mlogloss:0.298036

random_state=42：**0.472576**

> train-mlogloss:0.07944 val-mlogloss:0.297976

random_state=8：**0.474908**

> train-mlogloss:0.072324 val-mlogloss:0.313219

random_state=0：**0.473606**

> train-mlogloss:0.061369 val-mlogloss:0.285387

- ngram(ngram_range(1, 3))固定，random_state=42,修改subsample

subsample=0.8：0.472576

> train-mlogloss:0.07944 val-mlogloss:0.297976

subsample=0.7：0.474967

> train-mlogloss:0.073961 val-mlogloss:0.297218

subsample=0.9：0.472149

> train-mlogloss:0.085089 val-mlogloss:0.296884

subsample=1.0：0.471710

> train-mlogloss:0.0727 val-mlogloss:0.300373

- ngram(ngram_range(1, 3))、subsample=1固定，调随机种子

xgboost(1,3)_seed8_subsample1 

> train-mlogloss:0.079338 val-mlogloss:0.308202

xgboost(1,3)_seed0_subsample1

> train-mlogloss:0.080065 val-mlogloss:0.2925

xgboost(1,3)_seed4_subsample1

> train-mlogloss:0.088442 val-mlogloss:0.294975

xgboost(1,3)_seed42_subsample1

> train-mlogloss:0.0727 val-mlogloss:0.300373

四个随机种子结果取平均：**0.470195**

### 结果总结

1. ngram 模型效果优于 IFIDF 模型
2. ngram_range（1, 3）效果优于（1, 1）和（1, 5）
3. ngram_range 从 1 开始 优于 从 2 开始
4. min_df 和 max_df 参数修改后效果变差
5. subsample 取 1 效果最好
6. 4 个随机种子结果平均效果更好