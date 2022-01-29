# lbl_rf.py
from sklearn import ensemble
from sklearn import linear_model
from sklearn import preprocessing
import pandas as pd

def run(fold: int):
    # サンプルのCSVファイルをpandasに読み込む
    df = pd.read_csv("./train_folds.csv")
    
    # 
    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    
    ]
    
    # 個々の特徴量についてのループ
    for col in features:
        # 欠損値を文字列で補完し、全ての値を文字列に置換
        # 置換後の文字列をラベル＝ラベルIDとして使用する
        df.loc[:, col] = df[col].fillna("NONE").astype(str).values
        
    # 特徴量のラベルエンコーディング
    for col in features:
        # 初期化
        lbl = preprocessing.LabelEncoder()
        
        # ラベルエンコーダの学習
        lbl.fit(df[col])
        
        # データセットの変換
        df.loc[:, col] = lbl.transform(df[col])
        
    # 引数と一致しないkfoldのデータを学習
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # 引数と一致するkfoldのデータを検証
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # 学習データセットを準備
    x_train = df_train[features].values
    
    # 検証データセットを変換
    x_valid = df_valid[features].values
        
    # モデルの初期化と学習
    model = ensemble.RandomForestClassifier(n_jobs=1)
    model.fit(x_train, df_train.target.values)
    
    
    #  検証データセットを予測
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # AUCを算出
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    print(auc)
    
if __name__== "__main__":
    for fold_ in range(5):
        run(fold_)