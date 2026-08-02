import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 1.导入数据集

iris = load_iris()

# 2.划分训练集和验证集

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)

# 3.对数据进行标准化处理

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4.训练模型并进行预测

#默认所有“邻居”平权，每个人的投票一样重要
knn_uniform = KNeighborsClassifier(n_neighbors = 5)

#这里按邻居距离（ distance ）的远近来给他们的投票加权，距离越近，权重越大
knn_distance = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')

knn_uniform.fit(X_train, y_train)
knn_distance.fit(X_train, y_train)

y_pred_uniform = knn_uniform.predict(X_test)
y_pred_distance = knn_distance.predict(X_test)
print(f"实际结果：{y_test}")
print(f"uniform 预测结果：{y_pred_uniform}")
print(f"distance 预测结果：{y_pred_distance}")

# 5.评估模型效果

correct_uniform = np.sum(y_pred_uniform == y_test)
total = len(y_test)
acc_uniform = correct_uniform / total
print(f"uniform模型对验证集准确率为：{acc_uniform: .2%}")

correct_distance = np.sum(y_pred_distance == y_test)
acc_distance = correct_distance / total
print(f"distance模型对验证集准确率为：{acc_distance: .2%}")

# 6.决策边界可视化

# 取后两个特征（区分度最高），重新训练 + 标准化
feature_idx = [2, 3]  # petal length, petal width
X_2d = iris.data[:, feature_idx]

X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(
    X_2d, y, test_size = 0.2, random_state = 1
)

scaler_2d = StandardScaler()
X_train_2d = scaler_2d.fit_transform(X_train_2d)
X_test_2d = scaler_2d.transform(X_test_2d)

# 创建网格
h = 0.02  # 网格步长，步长越小，采样点越多，图像越平滑但绘制越慢
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 用不同 K 值画子图对比
k_values = [1, 5, 15, 30]
fig, axes = plt.subplots(2, 2, figsize = (12, 10))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold  = ['red', 'green', 'blue']

for ax, k in zip(axes.flat, k_values):
    knn_2d = KNeighborsClassifier(n_neighbors = k)
    knn_2d.fit(X_train_2d, y_train_2d)

    # 预测网格上每个点的类别，画决策背景
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap = cmap_light, alpha = 0.6)

    # 叠上训练点
    scatter = ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                         c = y_train_2d, cmap = ListedColormap(cmap_bold),
                         edgecolor = 'k', s = 50)

    acc = knn_2d.score(X_test_2d, y_test_2d)
    ax.set_title(f'K = {k}  |  测试集准确率: {acc:.2%}', fontsize=13)
    ax.set_xlabel(iris.feature_names[feature_idx[0]])
    ax.set_ylabel(iris.feature_names[feature_idx[1]])

plt.tight_layout()
plt.savefig('knn_decision_boundary.png', dpi = 300, bbox_inches = 'tight')