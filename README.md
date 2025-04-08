## 核心功能
- 支持两种特征提取模式：
  ```python
  # 高频词模式（原始实现）
  python classify.py --feature_mode=freq
  
  # TF-IDF模式
  python classify.py --feature_mode=tfidf
  ```
- 样本平衡处理（SMOTE）：
  ```python
  from imblearn.over_sampling import SMOTE
  smote = SMOTE(sampling_strategy={0: 127})
  ```

## 公式说明
- TF-IDF加权：  
  $$ \text{TF-IDF}(t,d) = \text{TF}(t,d) \times \log \frac{N}{1 + \text{DF}(t)} $$
- 朴素贝叶斯分类：  
  $$ \hat{y} = \argmax_y P(y) \prod_{i=1}^n P(x_i|y) $$
  原本代码部署

## 运行结果展示
![e3762bf7918325a9c209b58cafbd7c3](https://github.com/user-attachments/assets/af8e8bb4-bbd7-48a3-bdfe-76657f63873e)
实现特征选择方法的参数化切换机制，允许通过传入参数在以下两种特征提取方式间灵活选择：
![image](https://github.com/user-attachments/assets/f3249459-763e-48ad-9013-69a0617f0011)
![屏幕截图 2025-04-08 154415](https://github.com/user-attachments/assets/952a121a-58e4-4bc3-bdaa-c1fb32ceceb0)


## 选做任务完成
- 已完成样本平衡处理（SMOTE过采样）  
- 已增加精度/召回率/F1值评估报告  

### 5. 样本平衡处理
- 使用SMOTE过采样技术处理类别不平衡问题
- 原始样本分布：普通邮件24条，垃圾邮件127条
- 过采样后样本分布：普通邮件127条，垃圾邮件127条
- 采用k_neighbors=3的SMOTE参数配置

### 6. 模型评估指标
- 输出完整的分类评估报告，包括：
  - 精度(precision)
  - 召回率(recall) 
  - F1值
  - 支持数(support)
- 使用5折交叉验证评估模型稳定性
- 评估指标针对"普通邮件"和"垃圾邮件"两类分别计算
