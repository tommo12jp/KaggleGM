# create_folds.py
# StratifiedKFoldを使ってデータセットを分割する
import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    
    # サンプルのCSVファイルをpandasに読み込む
    df = pd.read_csv("./adult.csv")
    
    # kfold列を追加して初期化
    df["kfold"] = -1

    # サンプルをシャッフル
    df = df.sample(frac=1).reset_index(drop=True)

    # 目的変数の取り出し
    y = df.income.values
    
    # Startified K Foldクラスの初期化
    kf = model_selection.StratifiedKFold(n_splits=5)

    # kfold列を埋める
    for fold, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = fold
    
    # データセットを新しい名前で保存
    df.to_csv("./adult_folds.csv", index=False)