import pickle

import numpy as np
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lars
from sklearn.linear_model import LassoLars
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import ARDRegression
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error  # MSE
from sklearn.metrics import mean_absolute_error  # MAE
from sklearn.metrics import r2_score  # R 2
from eval import eval
from utils import train_test_split


# translate all data to .csv
def xlsx_to_csv_pd():
    data_xls = pd.read_excel("data.xlsx", index_col=0)
    data_xls.to_csv('data.csv', encoding='utf-8')


# normalize function z_score
def normalize_zscore(column):
    return (column - column.mean()) / column.std()


# normalize function min-max
def normalize_minmax(column):
    return (column - column.min()) / column.max() - column.min()




params = {
    'SGD': [{
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': [0.001, 0.1, 1, 10],
        'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5],
    }],
    'ElasticNet': [{
        'alpha': [0.0001, 0.001, 0.1, 1, 10],
        'l1_ratio': [0.96, 0.97, 0.98, 0.99, 1],
    }],
    'LassoLars': [{
        'alpha': [0.0001, 0.001, 0.1, 1, 10],
    }],
    'LassoLarsIC': [{
        'criterion': ['aic', 'bic'],
    }],
    'OMP': [{}],
    'ARD': [{}],
    'Bayes': [{}],
}

# estimators = [
#     ('LR', LinearRegression()),
#     ('ridge', RidgeCV()),
#
# )]

def fit_models(x_train, y_train):
    mfit = {model: models[model].fit(x_train, y_train) for model in models.keys()}
    b_params = {model: models[model].best_params_ for model in models.keys()}
    print(b_params)
    b_score = {model: models[model].best_score_ for model in models.keys()}
    print(b_score)
    return [(model, models[model].best_estimator_) for model in models.keys()]

def derive_positions(x_test, y_test, w, features):
    for model in models.keys():
        num = preds_result.shape[0]
        preds_result.loc[num] = [model, w, features, list(y_test), list(models[model].predict(x_test)), 0, 0, 0,
                                                   0]
        with open(f'./result/W{w} {model} {str(len(features))}features {num}.pickle', 'wb') as file:
            pickle.dump(models[model], file)

def array_to_string(arr):
    return ', '.join(map(str, arr))

def select_integrate(estimators, select_list):
    for ind in range(len(estimators))[::-1]:
        if estimators[ind][0] not in select_list:
            estimators.pop(ind)
    return estimators


if __name__ == "__main__":

    # translate all data to .csv
    # xlsx_to_csv_pd()

    # load all data
    df = pd.read_csv("data.csv")

    # remove some features
    # 'Adensity','AC_material_weight','TD_weight','Measure_Velocity','Overpressure'
    df = df.drop(columns=['Adensity', 'AC_material_weight', 'TD_weight', 'Measure_Velocity', 'Overpressure'])

    # normalize X
    data_norm = df.iloc[:, :-1].apply(normalize_zscore)
    data_norm['React_Efficiency'] = df['React_Efficiency']

    # feature selection
    X = data_norm.iloc[:, :-1].values
    y = data_norm['React_Efficiency'].values
    mi = mutual_info_regression(X, y)
    selector1 = SelectKBest(mutual_info_regression, k=10)
    selector2 = SelectKBest(mutual_info_regression, k=5)
    selector1.fit_transform(X, y)
    selector2.fit_transform(X, y)
    selected_features_idx_1 = selector1.get_support(indices=True)
    selected_features_idx_2 = selector2.get_support(indices=True)
    # Only Temp Rise
    selected_features_idx_3 = [14]
    preds_result = pd.DataFrame(columns=['model', 'W_ratio', 'features', 'gt', 'preds', 'MSE', 'MAE', 'RMSE', 'R2'])
    for selected_features_idx in [selected_features_idx_1, selected_features_idx_2, selected_features_idx_3]:
        # new data_norm after selected features

        selected_features_idx = list(selected_features_idx)
        selected_features_idx.append(15)
        s_data_norm = data_norm.iloc[:, selected_features_idx]
        # df1 = df.copy
        # s_data_norm.loc[:, 'React_Efficiency'] = df['React_Efficiency']

        # partition train_data and test_data according to the weight of w: [0, 10, 25, 50, 75]
        Ti_group = {0: [0, 6], 10: [6, 12], 25: [12, 19], 50: [19, 27], 75: [27, 34]}
        Zr_group = {0: [34, 40], 10: [40, 47], 25: [47, 52], 50: [52, 59], 75: [59, 65]}
        W_list = [0, 10, 25, 50, 75]

        for w in W_list:
            models = {
                # 'SGD': GridSearchCV(SGDRegressor(random_state=0, max_iter=100000), params['SGD'], cv=3),
                'ElasticNet': GridSearchCV(ElasticNet(random_state=0, max_iter=10000000), params['ElasticNet'], cv=3),
                'LassoLars': GridSearchCV(LassoLars(random_state=0, normalize=False), params['LassoLars'], cv=3),
                'LassoLarsIC': GridSearchCV(LassoLarsIC(normalize=False), params['LassoLarsIC'], cv=3),
                'OMP': GridSearchCV(OrthogonalMatchingPursuit(normalize=False), params['OMP'], cv=3),
                'ARD': GridSearchCV(ARDRegression(), params['ARD'], cv=3),
                'Bayes': GridSearchCV(BayesianRidge(), params['Bayes'], cv=3),
            }

            X_train, y_train, X_test, y_test = train_test_split(df, s_data_norm, w)

            print(f"--------------------w={w}--------------------")
            # integrate with best models
            estimators = fit_models(X_train, y_train)
            # select models
            estimators = select_integrate(estimators, ['ElasticNet', 'ARD', 'OMP', 'LassoLars'])
            derive_positions(X_test, y_test, w, features=X_train.columns.values)
            params["Stacking"] = [{
                'final_estimator': [ElasticNet(random_state=0, max_iter=1000000), None]
            }]
            models = {'Stacking': GridSearchCV(StackingRegressor(estimators=estimators), params["Stacking"], cv=3)}
            fit_models(X_train, y_train)
            derive_positions(X_test, y_test, w, features=X_train.columns.values)


        for ind, row in preds_result.iterrows():
            gt = row["gt"]
            x = range(len(gt))
            preds = row["preds"]
            w = row["W_ratio"]
            feature_num = str(len(row["features"]))

            preds_result.loc[ind, "MSE"] = mean_squared_error(gt, preds)
            preds_result.loc[ind, "MAE"] = mean_absolute_error(gt, preds)
            preds_result.loc[ind, "RMSE"] = np.sqrt(mean_squared_error(gt, preds))
            preds_result.loc[ind, "R2"] = r2_score(gt, preds)

            name = row["model"] + " " + feature_num + "feature"
            eval(x, gt, preds, name, w, save_fig=True)
        preds_result.to_excel('./result/model.xlsx')
            # preds_result.loc[i] = [k, w, x_axis, np.array(gt), np.array(preds)]
            # i += 1
            #
            # # predict by ensemble model
            # sclf = StackingRegressor(
            #     estimators=estimators,
            #     final_estimator=final_layer
            # )
            #
            # sclf.fit(X_train, y_train)
            # en_pred = sclf.predict(X_test)
            # eval(x, gt, en_pred, 'Ensemble', w, save_fig=False)

        #     preds_result.loc[i+1] = ['Ensemble', w, x_axis, np.array(gt), np.array(en_pred)]
        # columns_to_transform = ['x', 'gt', 'preds']
        # for col in columns_to_transform:
        #     preds_result[col] = preds_result[col].apply(lambda x: ', '.join(map(str, x)))
        #
        # preds_result.to_excel('result/preds.xlsx', index=False)

        """
        # X_tmp = np.arange(300, 2000, 1)
        # X_test = normalize_zscore(X_tmp).reshape(-1, 1)
    
        # interp_func = interp1d(X_tmp, preds, kind='linear')
        # X_tmp = np.linspace(min(X_tmp), max(X_tmp), 100)
        # y_interp = interp_func(X_tmp)
        # _, ax = plt.subplots()
        # ax.plot(X_tmp, y_interp)
        # plt.show()
        """






