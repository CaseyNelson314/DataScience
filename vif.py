import numpy as np
import pandas as pd
import japanize_matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


def vif(df):
    inverted_corr = np.linalg.inv(df.corr().values) # 相関行列の逆行列
    vif = np.diag(inverted_corr) # 対角成分を取得
    return pd.DataFrame(vif, index=df.columns, columns=["VIF"])

df = pd.read_csv("StudentPerformance.csv")

x = df[["前回成績", "勉強時間", "睡眠時間", "サンプル問題用紙枚数", "課外活動" ]]

x_dummied = pd.get_dummies(x, drop_first=True)

x_vif = vif(x_dummied)

sns.barplot(x=x_vif.index, y=x_vif["VIF"]).xaxis.set_label_text("")
plt.show()
