# lbl_xgb.py
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import xgboost as xgb

def run(fold: int):
    # サンプルのCSVファイルをpandasに読み込む
    df = pd.read_csv("./adult_folds.csv")
    
    # 数値を含む列の削除
    num_cols = [
        "age",
        "fnlwgt",
        #"educational-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week"
    ]
    df = df.drop(num_cols, axis=1)
    
    # 目的変数を0と1に置換
    target_mapping = {
        "<=50K": 0,
        ">50K": 1
    }
    df.loc[:, "income"] = df.income.map(target_mapping)
    
    features = [
        f for f in df.columns if f not in ("income", "kfold")
    
    ]
    
    # 特徴量のエンコーディング
    for col in features:
        # 欠損値を文字列で補完し、全ての値を文字列に置換
        # 置換後の文字列をラベル＝ラベルIDとして使用する
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # 特徴量のラベルエンコーディング
    for col in features:
        # 初期化
        lbl = preprocessing.LabelEncoder()
        
        # ラベルエンコーダの学習
        lbl.fit(df[col])
        
        # データセットの変換
        df.loc[:, col] = lbl.transform(df[col])
        
        
    # 引数と一致しないkfoldのデータを学習データにする
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # 引数と一致するkfoldのデータを検証データにする
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    
    # 学習データセットを準備
    x_train = df_train[features].values
    
    # 検証データセットを変換
    x_valid = df_valid[features].values
        
    # モデルの初期化と学習
    model = xgb.XGBClassifier(
        n_jobs=-1
    )
    model.fit(x_train, df_train.income.values)
    
    
    #  検証データセットを予測
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # AUCを算出
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    
    print(auc)
    
if __name__== "__main__":
    for fold_ in range(5):
        run(fold_)