import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    
    # サンプルのCSVファイルをpandasに読み込む
    df = pd.read_csv("/home/jovyan/work/project/input/mnist_train.csv")
    
    # kfold列を作って初期化
    df["kfold"] = -1

    # サンプルをシャッフル
    df = df.sample(frac=1).reset_index(drop=True)

    # Startified K Foldクラスの初期化
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)


    # kfold列を埋める
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y=df.label.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx, 'kfold'] = fold
    

    df.to_csv("/home/jovyan/work/project/input/mnist_train_folds.csv", index=False)