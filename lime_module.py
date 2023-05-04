from lime import lime_tabular
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

class limeExplainer():

    def __init__(self, x, features_df, model):
        self.data = x
        self.features = features_df
        self.mdl = model

    def return_weights(self, exp):
        exp_list = exp.as_map()[1]
        exp_list = sorted(exp_list, key=lambda x: x[0])
        exp_weight = [x[1] for x in exp_list]

        return exp_weight

    def get_features_sorted(self):
        explainer = lime_tabular.LimeTabularExplainer(self.data, feature_names=self.features)
        weights = []
        for instance in range(len(self.data)):
            exp = explainer.explain_instance(self.data[instance], self.mdl.predict_proba, num_features=len(self.features))

            # Get weights
            exp_weight = self.return_weights(exp)
            weights.append(exp_weight)

        lime_weights = pd.DataFrame(data=weights, columns=self.features)
        self.lime_weights = lime_weights

        # Get abs mean of LIME weights
        abs_mean = lime_weights.abs().mean(axis=0)
        abs_mean = pd.DataFrame(data={'feature': abs_mean.index, 'abs_mean': abs_mean})
        abs_mean = abs_mean.sort_values('abs_mean')

        self.abs_mean = abs_mean

        return list(abs_mean['feature'])


    def limeWeights_beeswarm(self, path_to_save):

        # BEESWARM
        X_df = pd.DataFrame(self.data)
        X_df.columns = self.variables

        plt.figure()

        # Use same order as mean plot
        y_ticks = range(len(self.abs_mean))
        y_labels = self.abs_mean.feature

        # plot scatterplot for each feature
        for i, feature in enumerate(y_labels):
            feature_weigth = self.lime_weights[feature]
            feature_value = X_df[feature][0:len(self.data)]

            plt.scatter(x=feature_weigth,
                        y=[i] * len(feature_weigth),
                        c=feature_value,
                        cmap='bwr',
                        edgecolors='black',
                        alpha=0.8)

        plt.vlines(x=0, ymin=0, ymax=9, colors='black', linestyles="--")
        plt.colorbar(label='Feature Value', ticks=[])

        plt.yticks(ticks=y_ticks, labels=y_labels, size=15)
        plt.xlabel('LIME Weight', size=20)

        """save..."""

