import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

data = pd.read_csv('inningScore.csv')

# sum elements in all_inning if it is not NaN
all_inning = [str(i) for i in range(1, 20)]
# print(all_inning)

data[all_inning] = data[all_inning].replace({np.nan: 0, '-': 0})

# Convert columns to numeric type
data[all_inning] = data[all_inning].apply(pd.to_numeric, errors='coerce')

data["final_score"] = data[all_inning].sum(axis=1, numeric_only=True)

# calculate the difference between the two teams per game
data['ScoreDiff'] = data.groupby('Game')['final_score'].transform(lambda x: x.diff())

ScoreDiff_data = data[['Game', 'ScoreDiff']].dropna()

plt.title('ScoreDiff to the end of the game')
sns.histplot(ScoreDiff_data['ScoreDiff'], bins=50, kde=True)

plt.show()


ScoreDiff_data = ScoreDiff_data['ScoreDiff'].values

ScoreDiff_data = ScoreDiff_data.reshape(-1, 1)

print(ScoreDiff_data)

# 判斷雙峰模型是否比單峰模型更好
gmm1 = GaussianMixture(n_components=1)
gmm1.fit(ScoreDiff_data)

gmm2 = GaussianMixture(n_components=2)
gmm2.fit(ScoreDiff_data)

print("Single component AIC:", gmm1.aic(ScoreDiff_data))
print("Two component AIC:", gmm2.aic(ScoreDiff_data))




param_grid = {
    'n_components': [2],  # 固定為雙峰模型
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],  # 探索不同的協方差類型
    'n_init': [1, 5, 10],  # 探索不同的初始化次數
    'random_state': [42]  # 設置隨機種子以保證結果可重現
}

# 創建GaussianMixture模型
gmm = GaussianMixture()

# 使用GridSearchCV進行參數搜索
search = GridSearchCV(gmm, param_grid, cv=5)  # 5-fold cross-validation
search.fit(ScoreDiff_data)

# 最佳參數
best_params = search.best_params_
print("Best parameters found: ", best_params)

# 使用最佳參數進行最終擬合
best_gmm = GaussianMixture(**best_params)
best_gmm.fit(ScoreDiff_data)

# 顯示最終模型的均值和協方差
print("Means: ", best_gmm.means_)
print("Covariances: ", best_gmm.covariances_)

# 計算標準差
standard_deviations = np.sqrt(best_gmm.covariances_ if best_params['covariance_type'] != 'tied' else np.array([best_gmm.covariances_]))
print("Standard deviations: ", standard_deviations)
