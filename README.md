原本代码部署
![e3762bf7918325a9c209b58cafbd7c3](https://github.com/user-attachments/assets/af8e8bb4-bbd7-48a3-bdfe-76657f63873e)
实现特征选择方法的参数化切换机制，允许通过传入参数在以下两种特征提取方式间灵活选择：
![image](https://github.com/user-attachments/assets/f3249459-763e-48ad-9013-69a0617f0011)
![屏幕截图 2025-04-08 151710](https://github.com/user-attachments/assets/e1812f81-7b3d-4abe-97fd-157c4773ce8b)
## 选做任务完成情况

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
