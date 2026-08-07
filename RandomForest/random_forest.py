import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1.读取数据集并获得其基本信息
df = pd.read_csv('train_and_test2.csv')
print(df.columns.to_list())

# 2.进行数据预处理
features_target = ['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked', '2urvived']
data = df[features_target]
nan = data.isnull().sum()
most_frequent_embarked = data['Embarked'].mode()[0]
data['Embarked'] = data['Embarked'].fillna(most_frequent_embarked, inplace = False)
print(data.describe())

# 3.对数据集进行划分
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 4.训练模型并进行预测
#通常来说树的数量（n_estimator）都是100起步，但由于本例中训练集样本过少，树过多反倒不一定有好处
#其他参数都和决策树一样
rf = RandomForestClassifier(random_state = 0, oob_score = True)     # oob_score 项用来设定是否需要计算该分数
param_grid = {'n_estimators': [15, 25, 50],
              'max_depth': [None, 5, 10, 15],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 3, 5],
              'criterion': ['gini', 'entropy'],
              'max_features': ['sqrt', None]
              }
grid_searcher = GridSearchCV(estimator = rf,
                             param_grid = param_grid,
                             cv = 5,    #做 5 次交叉验证
                             scoring = 'accuracy',
                             n_jobs = -1    #使用所有 cpu 核心进行并行运算
                             )

grid_searcher.fit(X_train, y_train)
print(f"最佳参数：{grid_searcher.best_params_}")
print(f"最佳交叉验证分数：{grid_searcher.best_score_}")

best_rf = grid_searcher.best_estimator_
y_pred = best_rf.predict(X_test)

# 5.评估模型训练效果
# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print(cm)
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

# accuracy, recall, precision, f1-score
print(classification_report(y_test, y_pred))

print(f"OOB准确率：{best_rf.oob_score_}")

from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

# 6.训练结果可视化
# ROC 曲线
RocCurveDisplay.from_estimator(best_rf, X_test, y_test)
plt.savefig('rf_roc_curve.png')

# 混淆矩阵热力图
ConfusionMatrixDisplay.from_estimator(best_rf, X_test, y_test, cmap='Blues')
plt.savefig('rf_confusion_matrix.png')