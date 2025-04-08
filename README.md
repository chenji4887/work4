原本代码部署
![e3762bf7918325a9c209b58cafbd7c3](https://github.com/user-attachments/assets/af8e8bb4-bbd7-48a3-bdfe-76657f63873e)
实现特征选择方法的参数化切换机制，允许通过传入参数在以下两种特征提取方式间灵活选择：
![image](https://github.com/user-attachments/assets/f3249459-763e-48ad-9013-69a0617f0011)
## 特征模式切换方法

### 1. 参数说明
通过 `--feature_method` 参数选择特征提取方法：
```bash
--feature_method high_freq   # 高频词模式（默认）
--feature_method tfidf       # TF-IDF加权模式
```

### 2. 技术细节对比
| 特征方法       | 实现原理                           | 特征维度 | 数据格式       |
|----------------|-----------------------------------|----------|----------------|
| 高频词(`high_freq`) | 选择训练集出现频率最高的3000个词 | 3000     | 词频统计向量   |
| TF-IDF(`tfidf`)    | 计算词频-逆文档频率权重           | 3000     | TF-IDF加权矩阵 |

### 3. TF-IDF公式说明
特征值计算公式：
$$
\text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log\left(\frac{1 + N}{1 + \text{DF}(t)}\right) + 1
$$
- $N$: 总文档数
- $\text{DF}(t)$: 包含词$t$的文档数
