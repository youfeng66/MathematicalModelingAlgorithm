import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1.导入数据集
df = pd.read_csv('breast_cancer.csv')
print(df)

# 2.划分训练集和测试集
X = df.iloc[:, :-1].values  #保留处最后一列 target 列意外的特征列
y = df.iloc[:, -1].values   #切除最后一列，即 target 列
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# 3.数据标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4.模型训练与预测
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

#获取一个二位列表，第一列为各样本判断为0的概率，第二列为各样本判断为1的概率
y_prob = model.predict_proba(X_test)
print(y_test)
print(y_pred)

#创建一个包含各样本判断结果及判断为1的概率值的 DataFrame
prob = pd.DataFrame(y_pred, columns = ['Predicted'])
prob['Probability'] = y_prob[:, 1]
prob.to_csv("预测结果.csv", index = False)

# 5.评估预测结果
print(classification_report(y_test, y_pred))

# 6.预测结果可视化
# ROC 曲线
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.savefig('lr_roc_curve.png')

# 混淆矩阵热力图
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap='Blues')
plt.savefig('lr_confusion_matrix.png')