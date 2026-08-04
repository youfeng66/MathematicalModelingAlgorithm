import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt


# 1.读取数据集并检查其数据分布状况
df = pd.read_csv('spam.csv', encoding = 'cp1252')
df['v1'] = df['v1'].replace({'spam' : 1, 'ham' : 0})

spam_count = np.sum(df['v1'] == 1)
total = df['v1'].count()
spam_proportion = spam_count / total
print(f'垃圾邮件在数据集的占比为：{spam_proportion:.2%}')
print(f'正常邮件在数据集的占比为：{1 - spam_proportion:.2%}')

# 2.对数据集进行划分
X = df['v2']
y = df['v1'].apply(lambda x: np.array(x, dtype = np.int32)).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# 3.数据预处理
vectorizer = CountVectorizer(
    encoding = 'cp1252',        # 文本解码方式和导入数据时的方式相同
    lowercase = True,           # 文本统一化小写
    stop_words = 'english',     # 去英语停用词
    ngram_range = (1, 2),       # 一元+二元词组
    max_df = 0.8,               # 过滤文档频率>80%的词
    min_df = 3,                 # 至少出现在5个文档中
)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# 4.训练模型并进行预测
model = MultinomialNB(
    alpha = 1.0,        # 数据平滑化处理
    fit_prior = True    # 让模型根据数据来调整两种类别的先验概率
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5.评估模型训练效果

# 混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

# accuracy, recall, precision, f1-score
print(classification_report(y_test, y_pred))

from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

# 6.模型预测效果可视化
# ROC 曲线
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.show('lr_roc_curve.png')

# 混淆矩阵热力图
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.show('lr_confusion_matrix.png')