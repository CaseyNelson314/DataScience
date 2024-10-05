#
#    重回帰分析
#

import pandas as pd
import japanize_matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# データ読み込み
df = pd.read_csv("StudentPerformance.csv")

# 目的変数と説明変数に分割
x = df[["前回成績", "勉強時間", "睡眠時間", "サンプル問題用紙枚数", "課外活動" ]]
y = df["成績"]

# ダミー変数化
x_dummied = pd.get_dummies(x, drop_first=True)

# 正規化
x_scaled = MinMaxScaler().fit_transform(x_dummied)
y_scaled = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))

# 重回帰分析
model = LinearRegression()
model.fit(x_scaled, y_scaled)

# 重みを表示
print(pd.DataFrame(model.coef_, columns=x_dummied.columns).T)
