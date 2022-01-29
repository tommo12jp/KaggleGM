# ohe_logres.py
import pandas as pd
from sklearn import metrics
from sklearn import linear_model
from sklearn import preprocessing

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
        
        # 引数と一致しないkfoldのデータを学習
        df_train = df[df.kfold != fold].reset_index(drop=True)
        
        # 引数と一致するkfoldのデータを検証
        df_valid = df[df.kfold == fold].reset_index(drop=True)
        
        # 初期化
        ohe = preprocessing.OneHotEncoder()
        
        
    #train-testの結合
    full_data = pd.concat([df_train[features], df_valid[features]],
                          axis=0
                         )
    #oheを学習
    ohe.fit(full_data[features])
    
    # 学習データセットを変換
    x_train = ohe.transform(df_train[features])
    
    # 検証データセットを変換
    x_valid = ohe.transform(df_valid[features])
        
    # モデルを初期化して学習
    model = linear_model.LogisticRegression()
    model.fit(x_train, df_train.target.values)
    
    #  検証データセットを予測
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # AUCを算出
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    print(auc)
    
if __name__== "__main__":
    for fold_ in range(5):
        run(fold_)