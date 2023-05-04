import warnings
warnings.filterwarnings("ignore", message=".*iteritems is deprecated.*")

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import xgboost as xgb
import shap
from lime import lime_tabular
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
from scipy.stats import ttest_ind
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import RocCurveDisplay, roc_auc_score, accuracy_score

path_to_fig = 'figures'
ROOT_PATH = '.'

seed = 42

datasets = ["Sha_CBSx_CD3.csv", "Sha_CBSx_CD3_Medulla.csv", "Sha_CBSx_CD11.csv", "Sha_CBSx_CD20.csv", "Sha_CBSx_Stroma.csv"]
outcomes = ['OS', 'PFS', 'POD12', 'POD24', 'RESP_ASSESS']

for dataset in datasets:
    for outcome in outcomes:
        df = pd.read_csv(os.path.join('datasets', dataset))

        #NA management
        df.loc[:, df.columns != 'NAME'] = df.loc[:, df.columns != 'NAME'].apply(pd.to_numeric, errors='coerce')

        #signing util numbers
        num_patients = df.shape[1] - 1
        num_features = df.shape[0] - len(outcomes)
        features_list = list(df['NAME'][0:num_features])

        #transpose dataframe and convert to matrix dropping outcomes NAs
        matrix = df.to_numpy()
        transpose = matrix.T
        dft = pd.DataFrame(transpose[1:num_patients], columns=df['NAME'])
        dft = dft.astype(float)
        dft = dft.dropna(subset=[outcome])
        num_patients = dft.shape[0]
        transpose = dft.to_numpy()
        X_tot = transpose[:, 0:num_features]
        Y_tot = np.array(dft[outcome], dtype=int)

        #splitting dataset
        X_train, X_test, y_train, y_test = train_test_split(X_tot, Y_tot,
                                                            test_size=0.3, random_state=seed, stratify=Y_tot)

        #imputer for NA management
        imputer = KNNImputer()
        imputer.fit(X_train)
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        #create Dmatrix
        dtrain = xgb.DMatrix(X_train, feature_names=features_list)
        dtest = xgb.DMatrix(X_test, feature_names=features_list)

        #setting XGBoost parameters
        param = {'max_depth': 5, 'eta': 0.001, 'objective': 'binary:logistic', 'eval_metric': 'auc'}

        #training model
        num_round = 500
        model = xgb.XGBClassifier(**param, n_estimators=num_round)
        model.fit(X_train, y_train)

        #model = xgb.train(param, dtrain, num_round)

        #make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        auc_train = roc_auc_score(y_train, y_pred_train)
        auc_test = roc_auc_score(y_test, y_pred_test)
        accuracy_train = accuracy_score(y_train, y_pred_train.round())
        accuracy_test = accuracy_score(y_test, y_pred_test.round())

        #calculate accuracy
        print(f"Dataset: {dataset}, Outcome: {outcome}")
        print(f"Accuracy / training set: {accuracy_train}")
        print(f"Accuracy / test set: {accuracy_test}")
        print(f"AUC      / training set: {auc_train}")
        print(f"AUC      / test set: {auc_test}")

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
        feature_importance_shap = shap_df_abs.mean().sort_values(ascending=False).index.tolist()


        # #Explain features importance using LIME weights (TOO SLOW!)
        # from lime_module import limeExplainer
        #
        # LE = limeExplainer(X_train, features_list, model)
        #
        # feature_importance_lime = LE.get_features_sorted()


        #features selection using p-value
        dfp = pd.DataFrame(X_train, columns=features_list)
        dfp[outcome] = y_train
        p_values = {}
        for feat in features_list:
            p_values[feat] = ttest_ind(dfp[dfp[outcome] == 0][feat], dfp[dfp[outcome] == 1][feat])[1]
        sort_pval_dict = sorted(p_values.items(), key=lambda x: x[1])
        top_features_pval = [x[0] for x in sort_pval_dict]

        #features selection using logistic regression


        #evaluating features selection techniques increasing the number of features to consider
        num_features_to_consider = [1, 2, 3, 4, 5, 10, 20, 50, 70, 90, 100, 120, 150, 200]
        aucs_method_test = []
        aucs_method_train = []
        labels = ["shap", "pval"]
        for feature_importance_method in [feature_importance_shap, top_features_pval]:
            auc_method_test = []
            auc_method_train = []
            for n in num_features_to_consider:
                top_features_method = feature_importance_method[0:n]
                df_top_method = dft[top_features_method]
                X_tot = df_top_method.to_numpy()
                X_train, X_test, y_train, y_test = train_test_split(X_tot, Y_tot, test_size=0.3, random_state=3)
                imputer = KNNImputer()
                imputer.fit(X_train)
                X_train = imputer.transform(X_train)
                X_test = imputer.transform(X_test)

                # XGBoost
                dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True, feature_names=top_features_method)
                dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True, feature_names=top_features_method)
                model = xgb.train(param, dtrain, num_round)
                y_pred_train = model.predict(dtrain)
                y_pred_test = model.predict(dtest)

                # MLP
                # mlp = MLPClassifier(hidden_layer_sizes=(10,), batch_size=32, max_iter=500)
                # mlp.fit(X_train, y_train)
                # y_pred_test = mlp.predict(X_test)

                auc_method_train.append(roc_auc_score(y_train, y_pred_train))
                auc_method_test.append(roc_auc_score(y_test, y_pred_test))

            aucs_method_train.append(auc_method_train)
            aucs_method_test.append(auc_method_test)

        #plotting results
        for aucs_method, subset in zip([aucs_method_train, aucs_method_test], ["train", "test"]):
            fig, ax = plt.subplots()
            for auc_m, label in zip(aucs_method, labels):
                ax.plot(num_features_to_consider, auc_m, '-o', label=label)
            ax.set_xlabel('num. features')
            ax.set_ylabel('AUC')
            ax.set_ylim([0, 1])
            ax.set_xticks(num_features_to_consider)
            plt.legend()
            path_to_shap_fig = os.path.join(path_to_fig, f"{subset}_plot_comparison_{outcome}_{dataset}.png")
            plt.tight_layout()
            plt.savefig(path_to_shap_fig)
            plt.close()
