import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

data = pd.read_csv('科技创新能力评价数据.csv')
data.set_index('地区', inplace = True)

regions = data.index.tolist()
columns = data.columns.tolist()
X_raw = data.values.astype(np.float64)
m, n = X_raw.shape
print(f"方案书 m = {m}, 指标数 n = {n}")

#1.指标正向化

X_pos = X_raw.copy()

#1.1成本型数据（GDP能耗）
#x = max(x) - x
col_cost = 2
X_pos[:, col_cost] = np.max(X_pos[:, col_cost], axis = 0) - X_pos[:, col_cost]

#1.2中间型数据（科技成果转化率）
#x = 1 - |x - x_best| / max| x_i - x_best|
col_mid = 5
x_best = 60
deviations = np.abs(X_pos[:, col_mid] - x_best)
max_dev = np.max(deviations)
if max_dev == 0:
    max_dev = 1
X_pos[:, col_mid] = 1 - deviations / max_dev

#1.3区间型数据（政策支持度）
#x 在[a, b]内，x = 1
#x < a , x = (a - x)/max(a - min(x), max(x) - b)
#x > b , x = (x - b)/max(a - min(x), max(x) - b)
col_interval = 6
a, b = 70, 90
vals = X_raw[:, col_interval]
M = max(a - vals.min(), b - vals.max())

new_vals = np.ones(vals.shape)
for i in range(m):
    if vals[i] < a:
        new_vals[i] = (a - vals[i]) / M
    elif vals[i] > b:
        new_vals[i] = (vals[i] - b) / M
X_pos[:, col_interval] = new_vals

print(pd.DataFrame(X_pos, index = regions, columns = columns))

#2.标准化

norm = X_pos / np.sqrt(np.sum(X_pos ** 2, axis = 0))
print(pd.DataFrame(norm.round(4), index = regions, columns = columns))

#3，配权重（熵权法）
# 3.1 先做 min-max 归一化（熵权法要求数据非负且同向）
norm_mm = (norm - norm.min(axis=0)) / (norm.max(axis=0) - norm.min(axis=0))
# 防止出现 0（log(0) = -inf）
norm_mm = np.clip(norm_mm, 1e-10, 1)

# 3.2 计算每个指标的信息熵
# P_ij = norm_mm_ij / sum(norm_mm_ij)  —— 第 i 方案在第 j 指标的占比
P = norm_mm / norm_mm.sum(axis=0)

# e_j = -k * sum(P_ij * ln(P_ij)), 其中 k = 1/ln(m)
k = 1 / np.log(m)
entropy = -k * np.sum(P * np.log(P), axis=0)   # 每个指标的信息熵

# 3.3 熵权：差异系数 / 总差异系数
diff_coeff = 1 - entropy                       # 信息效用值（差异系数）
weights = diff_coeff / diff_coeff.sum()        # 归一化得到权重

#4.TOPSIS计算

#4.1加权标准化矩阵
weighted = norm * weights

#4.2正理想解与负理想解
Z_plus = np.max(weighted, axis = 0)
Z_minus = np.min(weighted, axis = 0)

#4.3各对象到理想解的距离（欧几里得距离）
D_plus = np.sqrt(np.sum((weighted - Z_plus) ** 2, axis = 1))
D_minus = np.sqrt(np.sum((weighted - Z_minus) ** 2, axis = 1))

#4.4计算综合得分
scores = D_minus / (D_plus + D_minus)

max_score, index = np.max(scores), np.argmax(scores)
print(max_score.round(4), regions[int(index)])

sorted_index = scores.argsort()[:: -1]
rank = sorted_index.argsort() + 1
sorted_regions = np.array(regions)[sorted_index]
sorted_scores = scores[sorted_index]

#5.绘图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：得分柱状图（按省份排序）
axes[0].bar(sorted_regions, sorted_scores, color='steelblue', edgecolor='white')
axes[0].set_title('各省科技创新能力综合得分', fontsize=14)
axes[0].set_ylabel('TOPSIS 综合得分')
axes[0].set_ylim(0, 1)
for i, (r, s) in enumerate(zip(sorted_regions, sorted_scores)):
    axes[0].text(i, s + 0.01, f'{s:.3f}', ha='center', fontsize=8)

# 右图：权重饼图
wedges, texts, auto_texts = axes[1].pie(
    weights, labels=columns, autopct='%1.1f%%',
    colors=plt.cm.Set3(np.linspace(0, 1, 7)),
    startangle=90
)
axes[1].set_title('各指标权重（熵权法）', fontsize=14)

plt.tight_layout()
plt.savefig('topsis_result.png', dpi=200)
plt.show()
print("\n结果图已保存为 topsis_result.png")

#6.稳定性分析

stable_count = np.zeros(m)

for col_index in range(n):
    for delta in [-0.05, 0.05]:
        #微调权重
        w_test = weights.copy()
        w_test[col_index] += delta
        w_test = np.clip(w_test, 0.01, None)
        w_test /= w_test.sum()

        #重新计算Topsis
        weighted_test = w_test * norm
        Z_p = np.max(weighted_test, axis = 0)
        Z_m = np.min(weighted_test, axis = 0)
        D_p = np.sqrt(np.sum((weighted_test - Z_p) ** 2, axis = 1))
        D_m = np.sqrt(np.sum((weighted_test - Z_m) ** 2, axis = 1))
        scores_test = D_m / (D_p + D_m)

        #看排名变化情况
        rank_test = scores_test.argsort()[:: -1].argsort() + 1
        stable_count += (rank_test == rank)
stability = stable_count / (2 * n)

evaluation = []
for rate in stability:
    if rate >= 0.8:
        evaluation.append("高度稳定")
    elif rate >= 0.5:
        evaluation.append("较稳定")
    else:
        evaluation.append("不稳定")

df = pd.DataFrame({
    '地区' : regions,
    '排名' : rank,
    '稳定性' : stability,
    '评价' : evaluation
})

df.to_csv('各省排名稳定性分析.csv', index = False, encoding = 'utf-8')
print("各省排名的稳定性状况已保存为'各省排名稳定性分析.csv'")