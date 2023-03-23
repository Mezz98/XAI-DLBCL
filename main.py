import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
import shap

datasets = ["chapuy_example.csv"]
outcomes = ['OS', 'PFS']

for dataset in datasets:
    for outcome in outcomes:
        df = pd.read_csv(dataset)

N = df.shape[1]
num_features = df.shape[0] - len(outcomes)
features_list = list(df['NAME'])
matrix = df.to_numpy()
transpose = matrix.T
X_tot = transpose [1:N, 0:num_features]
dft = pd.DataFrame(transpose[1:N], columns=features_list[0:num_features+len(outcomes)])
Y_tot = list(dft[outcome])
X_train, X_test, y_train, y_test = train_test_split(X_tot, Y_tot, test_size=0.3, random_state=3)

#imputer

#Dmatrix
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# # Imposta i parametri per XGBoost
# param = {'max_depth': 3, 'eta': 0.1, 'objective': 'binary:logistic', 'eval_metric': 'auc'}
#
# # Addestra il modello
# num_round = 100
# bst = xgb.train(param, dtrain, num_round)
#
# # Fai le predizioni sul test set
# y_pred = bst.predict(dtest)
#
# # Calcola l'accuracy del modello
# accuracy = sum(y_pred.round() == y_test) / len(y_test)
# print(f"Accuracy: {accuracy}")
#
# # Explain feature importance using SHAP values
# explainer = shap.Explainer(model)
# shap_values = explainer(X)
#
# # Select top features based on SHAP values
# shap_df = pd.DataFrame(shap_values.values, columns=X.columns)
# shap_df_abs = shap_df.abs()
# feature_importance = shap_df_abs.mean().sort_values(ascending=False)
# top_features = feature_importance[:5].index.tolist()
#
# # Select only top features
# X_top = X[top_features]



#IMPLEMENTA ALTRO METODO DI FEATURES SELECTION

#CLASSIFICATORE ANN

#PRINT ACC INCREASING FEAT_NUM (COMPARISON FEAT_SEL TECHNIQUES)