import re
import os
from jieba import cut
from itertools import chain
from collections import Counter
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


class EmailClassifier:
    def __init__(self, feature_type='frequency', top_num=100):
        """
        初始化邮件分类器

        参数:
        - feature_type: 'frequency' 或 'tfidf'，指定特征选择方式
        - top_num: 当使用frequency时，选择的高频词数量
        """
        self.feature_type = feature_type
        self.top_num = top_num
        self.top_words = None
        self.vectorizer = None
        self.model = MultinomialNB()

    @staticmethod
    def get_words(filename):
        """读取文本并过滤无效字符和长度为1的词"""
        words = []
        with open(filename, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                # 过滤无效字符
                line = re.sub(r'[.【】0-9、——。，！~\*]', '', line)
                # 使用jieba.cut()方法对文本切词处理
                line = cut(line)
                # 过滤长度为1的词
                line = filter(lambda word: len(word) > 1, line)
                words.extend(line)
        return words

    def get_top_words(self, file_list):
        """遍历邮件建立词库后返回出现次数最多的词"""
        all_words = []
        for filename in file_list:
            all_words.append(self.get_words(filename))

        # 统计词频
        freq = Counter(chain(*all_words))
        return [i[0] for i in freq.most_common(self.top_num)]

    def preprocess_features(self, file_list):
        """根据选择的特征类型处理文本特征"""
        if self.feature_type == 'frequency':
            # 高频词特征
            self.top_words = self.get_top_words(file_list)
            vectors = []
            for filename in file_list:
                words = self.get_words(filename)
                word_map = list(map(lambda word: words.count(word), self.top_words))
                vectors.append(word_map)
            return np.array(vectors)

        elif self.feature_type == 'tfidf':
            # TF-IDF特征
            texts = []
            for filename in file_list:
                words = self.get_words(filename)
                texts.append(' '.join(words))

            self.vectorizer = TfidfVectorizer(max_features=self.top_num)
            return self.vectorizer.fit_transform(texts).toarray()

        else:
            raise ValueError("不支持的feature_type，请选择'frequency'或'tfidf'")

    def train(self, file_list, labels):
        """训练分类模型"""
        # 特征提取
        X = self.preprocess_features(file_list)

        # 训练模型
        self.model.fit(X, labels)

    def predict(self, filename):
        """预测邮件类别"""
        if self.feature_type == 'frequency':
            if self.top_words is None:
                raise RuntimeError("请先训练模型")
            words = self.get_words(filename)
            current_vector = np.array(
                tuple(map(lambda word: words.count(word), self.top_words)))
            result = self.model.predict(current_vector.reshape(1, -1))

        elif self.feature_type == 'tfidf':
            if self.vectorizer is None:
                raise RuntimeError("请先训练模型")
            words = self.get_words(filename)
            text = ' '.join(words)
            current_vector = self.vectorizer.transform([text]).toarray()
            result = self.model.predict(current_vector)

        else:
            raise ValueError("不支持的feature_type，请选择'frequency'或'tfidf'")

        return '垃圾邮件' if result == 1 else '普通邮件'


# 使用示例
if __name__ == "__main__":
    # 准备数据
    file_list = ['邮件_files/{}.txt'.format(i) for i in range(151)]
    labels = np.array([1] * 127 + [0] * 24)

    # 测试高频词特征
    print("=== 使用高频词特征 ===")
    classifier_freq = EmailClassifier(feature_type='frequency', top_num=100)
    classifier_freq.train(file_list, labels)

    # 测试TF-IDF特征
    print("\n=== 使用TF-IDF特征 ===")
    classifier_tfidf = EmailClassifier(feature_type='tfidf', top_num=100)
    classifier_tfidf.train(file_list, labels)

    # 测试预测
    test_files = ['邮件_files/{}.txt'.format(i) for i in range(151, 156)]
    for filename in test_files:
        print(f"\n{filename}分类情况:")
        print(f"高频词结果: {classifier_freq.predict(filename)}")
        print(f"TF-IDF结果: {classifier_tfidf.predict(filename)}")
