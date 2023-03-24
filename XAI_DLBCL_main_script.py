import warnings
warnings.filterwarnings("ignore", message=".*iteritems is deprecated.*")

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import xgboost as xgb
import shap
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from scipy.stats import ttest_ind
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import RocCurveDisplay

path_to_fig = 'figures'
ROOT_PATH = '.'

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
        dft = pd.DataFrame(transpose[1:N], columns=df['NAME'])
        dft = dft.astype(float)
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

        #Save SHAP summary plot figure
        plt.figure()
        shap.summary_plot(shap_values, X_train, feature_names=features_list)
        path_to_shap_fig = os.path.join(path_to_fig, "SHAP_summary_plot_{}_{}.png".format(outcome, dataset))
        plt.tight_layout()
        plt.savefig(path_to_shap_fig)
        plt.close()

        #Select top features based on SHAP values
        shap_df = pd.DataFrame(shap_values, columns=features_list)
        shap_df_abs = shap_df.abs()
        feature_importance_shap = shap_df_abs.mean().sort_values(ascending=False)

        #features selection using p-value
        dfp = pd.DataFrame(X_train, columns=features_list)
        dfp[outcome] = y_train
        p_values = {}
        for feat in features_list:
            p_values[feat] = ttest_ind(dfp[dfp[outcome] == 0][feat], dfp[dfp[outcome] == 1][feat])[1]
        sort_pval_dict = sorted(p_values.items(), key=lambda x: x[1])
        top_features_pval = [x[0] for x in sort_pval_dict]


        #evaluating features selection techniques increasing the number of features to consider
        num_features_to_consider = [1, 2, 3, 4]
        aucs_shap = []
        aucs_p = []
        for n in num_features_to_consider:
            top_features_shap = feature_importance_shap[0:n].index.tolist()
            df_top_shap = dft[top_features_shap]
            X_tot = df_top_shap.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X_tot, Y_tot, test_size=0.3, random_state=3)
            imputer = KNNImputer()
            imputer.fit(X_train)
            X_train = imputer.transform(X_train)
            X_test = imputer.transform(X_test)
            mlp = MLPClassifier(hidden_layer_sizes=(10,), batch_size=32, max_iter=500)
            mlp.fit(X_train, y_train)
            plt.figure()
            plt.ioff()
            viz = RocCurveDisplay.from_estimator(mlp, X_test, y_test)
            plt.close()
            aucs_shap.append(viz.roc_auc)

            top_features_pval_l = top_features_pval[0:n]
            df_top_pval = dft[top_features_pval_l]
            X_tot = df_top_pval.to_numpy()
            X_train, X_test, y_train, y_test = train_test_split(X_tot, Y_tot, test_size=0.3, random_state=3)
            imputer = KNNImputer()
            imputer.fit(X_train)
            X_train = imputer.transform(X_train)
            X_test = imputer.transform(X_test)
            mlp = MLPClassifier(hidden_layer_sizes=(10,), batch_size=32, max_iter=500)
            mlp.fit(X_train, y_train)
            plt.figure()
            plt.ioff()
            viz = RocCurveDisplay.from_estimator(mlp, X_test, y_test)
            plt.close()
            aucs_p.append(viz.roc_auc)

        #plotting results
        fig, ax = plt.subplots()
        ax.plot(num_features_to_consider, aucs_shap, '-o', label="shap")
        ax.plot(num_features_to_consider, aucs_p, '-o', label="p-value")
        ax.set_xlabel('num. features')
        ax.set_ylabel('AUC')
        ax.set_ylim([0, 1])
        plt.legend()
        path_to_shap_fig = os.path.join(path_to_fig, "plot_comparison_{}_{}.png".format(outcome, dataset))
        plt.tight_layout()
        plt.savefig(path_to_shap_fig)
        plt.close()