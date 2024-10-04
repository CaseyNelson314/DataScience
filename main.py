import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def vif(df):

    corr = df.corr()  # 相関行列
    inverted_corr = np.linalg.inv(corr.values) # 相関行列の逆行列
    vif = np.diag(inverted_corr) # 対角成分を取得

    return pd.DataFrame(vif, index=df.columns, columns=["VIF"])


# データ読み込み
df = pd.read_csv("StudentPerformance.csv")

# 目的変数と説明変数に分割
x = df[["前回成績", "勉強時間", "睡眠時間", "サンプル問題用紙枚数", "課外活動" ]]
y = df["成績"]

# ダミー変数化
x_dummied = pd.get_dummies(x, drop_first=True)

# VIF値を棒グラフに表示
x_vif = vif(x_dummied)
sns.barplot(x=x_vif.index, y=x_vif["VIF"]).xaxis.set_label_text("")
plt.show()

# 相関行列をヒートマップで表示
sns.heatmap(x_dummied.corr(), cmap="Reds", annot=True, fmt=".2f")
plt.show()

