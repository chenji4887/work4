# 邮件分类项目

## 核心功能
使用朴素贝叶斯分类器进行垃圾邮件识别，支持：
- 两种特征选择方法：高频词特征/TF-IDF加权特征
- 可选的SMOTE过采样处理（解决样本不平衡问题）
- 详细的分类评估报告（精度/召回率/F1值）

## 特征选择方法
### 1. 高频词特征
选择训练集中出现频率最高的N个词作为特征：
```python
特征向量 = [词1出现次数, 词2出现次数, ..., 词N出现次数]
```

### 2. TF-IDF特征
使用词频-逆文档频率加权：
$$
\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \left(\log\frac{1+N}{1+\text{DF}(t)} + 1\right)
$$
其中：
- $N$：总文档数
- $\text{DF}(t)$：包含词t的文档数

## 使用方法
```bash
# 基础模式
python classify.py --feature_method [high_freq|tfidf]

# 启用高级功能
python classify.py --feature_method tfidf --use_smote
```

## 评估指标示例
```text
              precision    recall  f1-score   support

           0       0.89      0.85      0.87        20
           1       0.97      0.98      0.98       125

    accuracy                           0.96       145
   macro avg       0.93      0.92      0.92       145
weighted avg       0.96      0.96      0.96       145
```# work4
