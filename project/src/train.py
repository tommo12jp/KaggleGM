import joblib
import pandas as pd
import os
import argparse
import config
import model_dispatcher

from sklearn import metrics
from sklearn import tree

def run(fold:int, model:str):
 
    # CSVファイルをdfに読み込み
    df = pd.read_csv(config.TRAINING_FILE)

    # 引数のfold番号と一致しないデータを学習に使う
    df_train = df[df.kfold!=fold].reset_index(drop=True)

    # 引数のfold番号と一致するデータを検証に使う
    df_test = df[df.kfold==fold].reset_index(drop=True)

    # 目的変数の列を削除し、numpy配列に変換（学習）
    X_train = df_train.drop("label", axis=1).values
    y_train = df_train.label.values

    # 目的変数の列を削除し、numpy配列に変換（検証）
    X_test = df_test.drop("label", axis=1).values
    y_test = df_test.label.values

    # model_dispatcherからモデルを取り出す
    clf = model_dispatcher.models[model]
    print(f"models: {model}")

    # モデルの学習
    clf.fit(X_train, y_train)

    # 検証用データに対する予測
    pred = clf.predict(X_test)

    # 正答率を算出
    accuracy = metrics.accuracy_score(y_test, pred)
    print(f"fold: {fold}, accuracy: {accuracy}")

    # モデル保存
    joblib.dump(clf, 
    os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin"))

if __name__ == "__main__":
    # argparseの初期化
    parser = argparse.ArgumentParser()
    
    # 引数と型を追加
    parser.add_argument(
        "--fold",
        type=int
    )
    parser.add_argument(
        "--model",
        type=str
    )
    
    
    # コマンドライン引数の読み込み
    args = parser.parse_args()
    
    # 引数で指定したfold番号について実行
    run(
        fold=args.fold,
        model=args.model
    )