# Solejay-code
Renjie Pu's code

### load_file.py

读取训练集和测试集数据，处理后保存每个文件对应的 `label` 和 `拼接的 api 序列` 并保存为 `pkl` 文件实现数据的持久存储。

### train.py

用 ngram 或 TFIDF 提取 api 序列特征，采用五折交叉验证用 xgboost 进行分类

### 提交结果

- 词袋模型+xgboost 5折交叉验证

unigram：0.567571（ngram_range=(1, 1)）

3_gram：0.473048（ngram_range=(1, 3)）

5_gram：0.473122（ngram_range=(1, 5)）

- TFIDF模型+xgboost 5折交叉验证

unigram：0.659438（ngram_range=(1, 1)）

3_gram：0.505826（ngram_range=(1, 3)）

5_gram：0.507149（ngram_range=(1, 5)）

