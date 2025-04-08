# 导入必要的库
from collections import Counter
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns


def generate_imbalanced_data():
    """生成模拟的不平衡数据集"""
    X, y = make_classification(
        n_samples=151,  # 总样本数
        n_features=10,  # 特征数量
        n_classes=2,  # 类别数
        weights=[0.2, 0.8],  # 类别比例（20% vs 80%）
        random_state=42  # 随机种子
    )
    return X, y


def analyze_class_distribution(y, title):
    """分析并可视化类别分布"""
    print(f"\n{title}类别分布:")
    print(f"类别0（少数类）: {sum(y == 0)}个样本")
    print(f"类别1（多数类）: {sum(y == 1)}个样本")

    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title(title)
    plt.xticks([0, 1], ['类别0（少数）', '类别1（多数）'])
    plt.show()


def apply_smote(X_train, y_train):
    """应用SMOTE过采样"""
    print("\n应用SMOTE前训练集分布:", Counter(y_train))
    sm = SMOTE(
        sampling_strategy='auto',  # 自动平衡到多数类数量
        k_neighbors=5,  # 生成样本时考虑的邻居数
        random_state=42
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print("应用SMOTE后训练集分布:", Counter(y_res))
    return X_res, y_res


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练模型并评估性能"""
    # 初始化随机森林分类器
    model = RandomForestClassifier(
        n_estimators=100,  # 决策树数量
        class_weight='balanced',  # 自动平衡类别权重
        random_state=42
    )

    # 训练模型
    model.fit(X_train, y_train)

    # 预测测试集
    y_pred = model.predict(X_test)

    # 输出分类报告
    print("\n分类性能报告:")
    print(classification_report(y_test, y_pred, target_names=['类别0', '类别1']))

    # 绘制混淆矩阵
    plt.figure(figsize=(6, 4))
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['预测0', '预测1'],
                yticklabels=['真实0', '真实1'])
    plt.title('混淆矩阵')
    plt.show()


def main():
    # 1. 生成不平衡数据
    X, y = generate_imbalanced_data()
    analyze_class_distribution(y, "原始数据集")

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,  # 20%测试集
        stratify=y,  # 保持类别比例
        random_state=42
    )
    analyze_class_distribution(y_train, "训练集原始分布")

    # 3. 应用SMOTE过采样
    X_res, y_res = apply_smote(X_train, y_train)

    # 4. 训练和评估模型
    train_and_evaluate(X_res, y_res, X_test, y_test)


if __name__ == "__main__":
    main()
