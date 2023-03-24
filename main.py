import warnings
warnings.filterwarnings("ignore", message=".*iteritems is deprecated.*")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import xgboost as xgb
import shap
import matplotlib
import matplotlib.pyplot as plt

datasets = ["chapuy_example.csv"]
outcomes = ['OS', 'PFS']

for dataset in datasets:
    for outcome in outcomes:
        df = pd.read_csv(dataset)

#NA management
df_temp = df.drop('NAME', axis=1)
df_temp = df_temp.apply(pd.to_numeric, errors='coerce')
df_temp.insert(0, 'NAME', df['NAME'])
df = df_temp
del df_temp

#signing util numbers
N = df.shape[1]
num_features = df.shape[0] - len(outcomes)
features_list = list(df['NAME'][0:6])

#transpose dataframe and convert to matrix dropping outcomes NAs
matrix = df.to_numpy()
transpose = matrix.T
dft = pd.DataFrame(transpose[1:N], columns=features_list[0:num_features+len(outcomes)])
dft = dft.dropna(subset=outcome)
N = dft.shape[1]
transpose = dft.to_numpy()
X_tot = transpose [:, 0:num_features]
Y_tot = list(dft[outcome])

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_tot, Y_tot, test_size=0.3, random_state=3)

#imputer for NA management
imputer = KNNImputer()
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

#create Dmatrix
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True, feature_names=features_list)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True, feature_names=features_list)

#setting XGBoost parameters
param = {'max_depth': 2, 'eta': 0.03, 'objective': 'binary:logistic', 'eval_metric': 'auc'}

#training model
num_round = 100
model = xgb.train(param, dtrain, num_round)

#make predictions
y_pred_train = model.predict(dtrain)
y_pred_test = model.predict(dtest)

#calculate accuracy
accuracy = sum(y_pred_train.round() == y_train) / len(y_train)
print(f"Accuracy / training set: {accuracy}")
accuracy = sum(y_pred_test.round() == y_test) / len(y_test)
print(f"Accuracy / test set: {accuracy}")

#Explain feature importance using SHAP values
shap_values = shap.TreeExplainer(model).shap_values(X_train)

shap.summary_plot(shap_values, X_train, feature_names=features_list)

#Select top features based on SHAP values
shap_df = pd.DataFrame(shap_values.values, columns=features_list)
shap_df_abs = shap_df.abs()
feature_importance = shap_df_abs.mean().sort_values(ascending=False)
top_features = feature_importance[:5].index.tolist()

#Select only top features
X_top = X[top_features]



#IMPLEMENTA ALTRO METODO DI FEATURES SELECTION

#CLASSIFICATORE ANN

#PRINT ACC INCREASING FEAT_NUM (COMPARISON FEAT_SEL TECHNIQUES)