import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("StudentPerformance.csv")

x = df[["前回成績", "勉強時間", "睡眠時間", "サンプル問題用紙枚数", "課外活動" ]]

x_dummied = pd.get_dummies(x, drop_first=True)  # ダミー変数化

sns.heatmap(x_dummied.corr(), cmap="Reds", annot=True, fmt=".2f")
plt.show()
