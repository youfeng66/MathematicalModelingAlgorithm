import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df = pd.read_csv('train_and_test2.csv')
print(df.columns.tolist())

features_target = ['Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass', 'Embarked', '2urvived']
data = df[features_target]
nan = data.isnull().sum()
print(f'各特征列数据确实情况为：\n{nan}')

most_frequent_embarked = data['Embarked'].mode()[0]
data['Embarked'] = data['Embarked'].fillna(most_frequent_embarked, inplace = False)

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

base_tree = DecisionTreeClassifier(
    criterion = 'gini',   #或 'entropy'
    max_depth = 10,
    min_samples_split = 5,
    min_samples_leaf = 3,
    max_features = 5,
    random_state = 42
)
base_tree.fit(X_train, y_train)
y_pred = base_tree.predict(X_test)

# 获取过拟合树的代价复杂度剪枝路径（获得一系列 alpha 值和对应的子树的叶子数、误差）
path = base_tree.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas  # 候选的 alpha 值（从小到大）
impurities = path.impurities  # 对应 alpha 的误差变化

test_scores = []

# 对每个 alpha 训练一棵剪枝后的树，并评估其在测试集上的表现
# 注意：ccp_alphas 的最后一个值很大，可能使树只剩下根节点，我们通常不取最后一个
for alpha in ccp_alphas[:-1]:  # 去掉最后一个，避免给枝干全剪完了
    tree = DecisionTreeClassifier(
    criterion = 'gini',
    max_depth = 10,
    min_samples_split = 5,
    min_samples_leaf = 3,
    max_features = 5,
    ccp_alpha=alpha,
    random_state = 10
    )
    tree.fit(X_train, y_train)
    test_scores.append(accuracy_score(y_test, tree.predict(X_test)))
best_alpha = ccp_alphas[:-1][np.argmax(test_scores)]
# 得到最终模型
final_tree = DecisionTreeClassifier(
    criterion = 'gini',
    max_depth = 10,
    min_samples_split = 5,
    min_samples_leaf = 3,
    max_features = 5,
    ccp_alpha = best_alpha,
    random_state = 10
)
final_tree.fit(X_train, y_train)
y_pred = final_tree.predict(X_test)

# 混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
print(cm)
TN, FP = cm[0, 0], cm[0, 1]
FN, TP = cm[1, 0], cm[1, 1]

# accuracy, recall, precision, f1-score
print(classification_report(y_test, y_pred))

# 树结构图
from sklearn.tree import plot_tree

plt.figure(figsize = (20, 15))
plot_tree(final_tree,
          feature_names = features_target[:-1],
          filled = False,
          rounded = True,
          fontsize = 14)

from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay

# ROC 曲线
RocCurveDisplay.from_estimator(final_tree, X_test, y_test)
plt.savefig('lr_roc_curve.png')

# 混淆矩阵热力图
ConfusionMatrixDisplay.from_estimator(final_tree, X_test, y_test, cmap='Blues')
plt.savefig('lr_confusion_matrix.png')