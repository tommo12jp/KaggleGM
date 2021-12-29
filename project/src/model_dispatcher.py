from sklearn import tree
from sklearn import ensemble

"""
複数のモデルタイプを引数の値に応じて返す
ジニ係数とエントロピーを基準んいする2つの決定木を準備
"""
models = {
    "decision_tree_gini": tree.DecisionTreeClassifier(
        criterion="gini"
    ),
    "decision_tree_entropy": tree.DecisionTreeClassifier(
        criterion="entropy"
    ),
    "rf": ensemble.RandomForestClassifier(),
}