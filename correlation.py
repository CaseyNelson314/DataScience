import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# データ読み込み
df = pd.read_csv("StudentPerformance.csv")

# 説明変数を抽出
x = df[["前回成績", "勉強時間", "睡眠時間", "サンプル問題用紙枚数", "課外活動" ]]

# ダミー変数化
x_dummied = pd.get_dummies(x, drop_first=True)  # ダミー変数化

# 相関係数を表示
sns.heatmap(x_dummied.corr(), cmap="Reds", annot=True, fmt=".2f")
plt.show()
