import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 1.导入数据集
diabetes = load_diabetes()
print(diabetes['DESCR'])

# 2.划分训练集和验证集
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 3.数据进行标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4.训练模型并进行预测
ridge = RidgeCV()
param_grid = {'alphas': [0.1, 1, 10, 100, 1000]}
grid_searcher = GridSearchCV(estimator = ridge,
                             param_grid = param_grid,
                             cv = 5,    #做 5 次交叉验证
                             scoring = 'accuracy',
                             n_jobs = -1    #使用所有 cpu 核心进行并行运算
                             )

grid_searcher.fit(X_train, y_train)
print(f"最佳参数：{grid_searcher.best_params_}")
print(f"最佳交叉验证分数：{grid_searcher.best_score_}")

best_ridge = grid_searcher.best_estimator_
y_pred = best_ridge.predict(X_test)

# 5.评估预测结果
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}  (越小越好)")
print(f"RMSE: {rmse:.2f}  (越小越好，和 y 同单位)")
print(f"MAE:  {mae:.2f}  (越小越好，对异常值不敏感)")
print(f"R²:   {r2:.3f}  (越接近 1 越好)")

# 6.预测结果可视化
# 6.1 预测值 vs 真实值 + 残差分布
residuals = y_test - y_pred

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 左图：预测值 vs 真实值
ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('真实值')
ax1.set_ylabel('预测值')
ax1.set_title('预测值 vs 真实值')

# 右图：残差分布
ax2.scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('预测值')
ax2.set_ylabel('残差')
ax2.set_title('残差分布图')

plt.tight_layout()
plt.savefig('ridge_pred_residual.png', dpi=200, bbox_inches='tight')

# 6.2 Alpha 路径图——系数收缩过程
alphas = np.logspace(-3, 3, 50)  # 0.001 到 1000，对数均匀取 50 个点
coefs = []
for a in alphas:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train, y_train)
    coefs.append(ridge.coef_)

plt.figure(figsize=(10, 6))
plt.plot(alphas, coefs)
plt.xscale('log')
plt.xlabel('alpha (正则化强度)')
plt.ylabel('系数值')
plt.title('不同 alpha 下的系数收缩路径')
plt.legend(diabetes.feature_names, bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.savefig('ridge_alpha_path.png', dpi=200, bbox_inches='tight')