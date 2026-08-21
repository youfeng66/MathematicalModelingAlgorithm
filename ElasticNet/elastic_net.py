import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# 1.导入并观察数据集的基本情况
diabetes = load_diabetes()
print(diabetes['DESCR'])
df = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
print(df)

# 2.划分训练集和验证集
X = diabetes.data
y = diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 3.数据进行标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4.训练模型并进行预测
modelCV = ElasticNetCV(
        alphas = np.logspace(-3, 1, 50),     # 候选 alpha
        l1_ratio = [.1, .3, .5, .7, .9, 1],  # 候选混合比例
        cv = 5,
        n_jobs = -1
        )
modelCV.fit(X_train, y_train)
y_pred = modelCV.predict(X_test)

# 5.对结果进行评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}  (越小越好)")
print(f"RMSE: {rmse:.2f}  (越小越好，和 y 同单位)")
print(f"MAE:  {mae:.2f}  (越小越好，对异常值不敏感)")
print(f"R²:   {r2:.3f}  (越接近 1 越好)")

# 6.预测结果可视化
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