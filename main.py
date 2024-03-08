import pickle

import numpy as np
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
import sklearn.model_selection as ms
from omnixai.data.tabular import Tabular
from omnixai.explainers.prediction import PredictionAnalyzer
from omnixai.explainers.tabular import TabularExplainer
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.preprocessing.base import Identity
# from omnixai.visualization.dashboard import Dashboard
# from omnixai.visualization.dashboard import Dashboard
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


from eval import *
from utils import train_test_split


# translate all data to .csv
def xlsx_to_csv_pd():
    data_xls = pd.read_excel("data.xlsx", index_col=0)
    data_xls.to_csv('data.csv', encoding='utf-8')


# normalize function z_score
def normalize_zscore(column):
    # print(column.mean(), column.std())
    # return (column - column.mean()) / column.std()
    return (column - 1500.0) / 865.7366805212772


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


def fit_models(models_list, x_train, y_train, verbose=False):
    mfit = {model: models_list[model].fit(x_train, y_train) for model in models_list.keys()}
    b_params = {model: models_list[model].best_params_ for model in models_list.keys()}
    b_score = {model: models_list[model].best_score_ for model in models_list.keys()}
    
    if verbose:
        print(b_score)
        print(b_params)
    
    print(f"In fit function: {id(models_list)}")
    return [(model, models_list[model].best_estimator_) for model in models_list.keys()]

def derive_positions(models_list, x_test, y_test, w, features, new_bee):
    for model in models_list.keys():
        num = preds_result.shape[0]
        preds_result.loc[num] = [model, w, features, list(y_test), list(models_list[model].predict(x_test)), new_bee, 0, 0, 0,
                                                   0]
        # with open(f'./result/W{w} {model} {str(len(features))}features {num}.pickle', 'wb') as file:
        #     pickle.dump(models_list[model], file)

def derive_prediction(models_list, x_test, w, features):
    print(f"In predict function: {id(models_list)}")
    # print(x_test.shape)
    for model in models_list.keys():
        num = preds_result.shape[0]
        preds_result.loc[num] = [model, w, features, None, list(models_list[model].predict(x_test)), 0, 0, 0,
                                                   0]

def array_to_string(arr):
    return ', '.join(map(str, arr))

def select_integrate(estimators, select_list):
    for ind in range(len(estimators))[::-1]:
        if estimators[ind][0] not in select_list:
            estimators.pop(ind)
    return estimators

# TODO:约定使用CEP_Q, CEP_q, Impact_Temp_Rise 三个特征，每种配方使用一组CEP_Q和CEP_q
#  task：针对训练好的模型，预测给定冲击温升在[300, 2000, 1]的反应效率，所以我把相应的CEP_Q和CEP_q
#  以及冲击温升做成了新的test_data
def generate_test(scope:list,
                  step:int,
                  Q:float,
                  q:float):
    start, stop = scope[0], scope[1]+1
    Impact_temp = np.arange(start, stop, step).reshape(-1, 1)
    # x = np.arange(start, stop, step).reshape(-1, 1)
    # Impact_temp = np.cos(x).reshape(-1, 1)
    Impact_temp = normalize_zscore(Impact_temp)
    CEP_Q = np.full_like(Impact_temp, Q, dtype=float)
    CEP_q = np.full_like(Impact_temp, q, dtype=float)
    X_test = np.concatenate([CEP_Q, CEP_q, Impact_temp], axis=1)
    # X_test = Impact_temp

    return X_test

def analysis(data,
             train_data,
             test_data,
             test_labels,
             model
             ):
    # TODO: 这里是将我们的数据转换为OMNIXAI的表格类。很简单，把我们预测的那一项放在target_columns就行
    tabular_data = Tabular(
        data,
        target_column='React_Efficiency'
    )
    # TODO: 这里一般是预处理。不用怎么变动。
    transformer = TabularTransform(
        target_transform=Identity()
    ).fit(tabular_data)
    class_names = transformer.class_names

    # 这里是将训练数据转换称为解释库的表格数据，这一步很重要，为了分析用。
    # Convert the transformed data back to Tabular instances
    train_data = transformer.invert(train_data.values)
    test_data = transformer.invert(test_data.values)

    # Initialize a TabularExplainer
    explainer = TabularExplainer(
        # TODO: 这里是解释器，需要去官网看看还有那些解释器可以用
        explainers=["lime", "shap", "ale"],  # The explainers to apply
        mode="regression",  # The task type
        data=train_data,  # The data for initializing the explainers
        model=model,  # The ML model to explain
        preprocess=lambda z: transformer.transform(z),  # Converts raw features into the model inputs
        # TODO: 传入解释器的参数。
        params=None
    )

    # TODO: 这里是用于展示在dashboard的测试样例，很重要，我这里只是随意的挑选，我们需要把要分析的数据放进来。
    test_instances = test_data[:5]
    # TODO: 这两个解释器的差别，我还没细看。你可以仔细看看
    local_explanations = explainer.explain(X=test_instances)
    global_explanations = explainer.explain_global(
    )

    # TODO: 后面的就是标配了，你也可以看看具体有哪些分析框架可以纳入。官方文档写的很详细
    analyzer = PredictionAnalyzer(
        mode="regression",
        test_data=test_data,  # The test dataset (a `Tabular` instance)
        test_targets=test_labels,  # The test labels (a numpy array)
        model=model,  # The ML model
        preprocess=lambda z: transformer.transform(z)  # Converts raw features into the model inputs
    )
    prediction_explanations = analyzer.explain()

    # Launch a dashboard for visualization
    # dashboard = Dashboard(
    #     instances=test_instances,  # The instances to explain
    #     local_explanations=local_explanations,  # Set the generated local explanations
    #     global_explanations=global_explanations,  # Set the generated global explanations
    #     prediction_explanations=prediction_explanations,  # Set the prediction metrics
    #     class_names=class_names,  # Set class names
    #     explainer=explainer  # The created TabularExplainer for what if analysis
    # )
    # dashboard.show(host='127.0.0.1', port='8888')  # Launch the dashboard
    # dashboard.show()


if __name__ == "__main__":

    # translate all data to .csv
    # xlsx_to_csv_pd()

    # load all data
    df = pd.read_csv("Data/data.csv")

    # remove some features
    # 'Adensity','AC_material_weight','TD_weight','Measure_Velocity','Overpressure'
    df = df.drop(columns=['Adensity', 'AC_material_weight', 'TD_weight', 'Measure_Velocity', 'Overpressure'])

    # normalize X
    data_norm = df.iloc[:, :-1].apply(normalize_zscore)
    # 600 800 900 1200
    # 600 - 1800 -> z
    # data_norm = df.iloc[:, :-1]
    data_norm['React_Efficiency'] = df['React_Efficiency']
    # print(data_norm.head(5))

    # feature selection
    X = data_norm.iloc[:, :-1].values
    y = data_norm['React_Efficiency'].values
    mi = mutual_info_regression(X, y)
    # selector1 = SelectKBest(mutual_info_regression, k=5)    # select 5 features
    # selector2 = SelectKBest(mutual_info_regression, k=3)
    # selector1.fit_transform(X, y)
    # selector2.fit_transform(X, y)
    # selected_features_idx_1 = selector1.get_support(indices=True)
    selected_features_idx_1 = [10, 11, 14]
    # selected_features_idx_2 = selector2.get_support(indices=True)
    # Only Temp Rise
    selected_features_idx_3 = [14]
    preds_result = pd.DataFrame(columns=['model', 'W_ratio', 'features', 'gt', 'preds', 'new_bee', 'MSE', 'MAE', 'RMSE', 'R2'])

    # for selected_features_idx in [selected_features_idx_2, selected_features_idx_3]:
    for selected_features_idx in [selected_features_idx_1]:

        # new data_norm after selected features
        selected_features_idx = list(selected_features_idx)
        selected_features_idx.append(15)
        s_data_norm = data_norm.iloc[:, selected_features_idx]
        # print(s_data_norm.head(5))
        # df1 = df.copy
        # s_data_norm.loc[:, 'React_Efficiency'] = df['React_Efficiency']

        # partition train_data and test_data according to the weight of w: [0, 10, 25, 50, 75]
        Ti_group = {0: [0, 6], 10: [6, 12], 25: [12, 19], 50: [19, 27], 75: [27, 34]}
        Zr_group = {0: [34, 40], 10: [40, 47], 25: [47, 52], 50: [52, 59], 75: [59, 65]}
        # W_list = [0, 10, 25, 50, 75]
        W_list = [[0, 10, 25, 50, 75],[0, 10, 25, 50, 75]]
        L_list = ["Ti", "Zr"]

        for l in range(len(L_list)):
            for w in W_list[l]:
                models = {
                    # 'SGD': GridSearchCV(SGDRegressor(random_state=0, max_iter=100000), params['SGD'], cv=3),
                    'ElasticNet': GridSearchCV(ElasticNet(random_state=0, max_iter=10000000), params['ElasticNet'], cv=3, n_jobs=-1),
                    'LassoLars': GridSearchCV(LassoLars(random_state=0, normalize=False), params['LassoLars'], cv=3, n_jobs=-1),
                    'LassoLarsIC': GridSearchCV(LassoLarsIC(normalize=False), params['LassoLarsIC'], cv=3, n_jobs=-1),
                    'OMP': GridSearchCV(OrthogonalMatchingPursuit(normalize=False), params['OMP'], cv=3, n_jobs=-1),
                    'ARD': GridSearchCV(ARDRegression(), params['ARD'], cv=3, n_jobs=-1),
                    'Bayes': GridSearchCV(BayesianRidge(), params['Bayes'], cv=3, n_jobs=-1),
                }
                print(f"--------------------w={str(w)+' '+ L_list[l]}--------------------")
                
                print(f"In init function: {id(models)}")

                # X_train, y_train, X_test, y_test, Impact_Temp_Rise = train_test_split(df, s_data_norm, l, w)
                # print(X_test.head())
                # X_test 和 X_train划分不改变，但是X_test 变更为 对单项材料冲击温升在[300,2000]范围内变化的预测
                X_train, y_train, value, y_test, new_bee = train_test_split(df, s_data_norm, l, w)

                # TODO 新的test dat
                X_test = generate_test([600, 1800], 1, value[0], value[1])
                columns = ["CEP_Q", "CEP_q", "Impact_Temp_Rise"]
                X_test = pd.DataFrame(X_test, columns=columns)
                # print(X_test)

                
                # integrate with best models
                estimators = fit_models(models, X_train, y_train)
                # select models
                estimators = select_integrate(estimators, ['ElasticNet', 'ARD', 'OMP', 'LassoLars'])
                # derive_positions(X_test, y_test, str(w)+' '+L_list[l], features=X_train.columns.values)
                
                # TODO: 训练好的模型放在这里。可以接一个接口进来。
                # with open(f'./result/W{"75"} {"Zr"} {"Stacking"} {"3"}features {"19"}.pickle', 'rb') as file:
                #     model = pickle.load(file)
                # analysis(s_data_norm, X_train, X_test, y_test, model)

                params["Stacking"] = [{
                    'final_estimator': [ElasticNet(random_state=0, max_iter=1000000), None]
                }]
                # new_models = {'Stacking': GridSearchCV(StackingRegressor(estimators=estimators), params["Stacking"], cv=3)}
                # new_models = {'Stacking': StackingRegressor(estimators=estimators, final_estimator=ElasticNet(random_state=0, max_iter=100))}
                nnn = [('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=1, max_iter=10000000, random_state=0)), ('LassoLars', LassoLars(alpha=0.1, normalize=False, random_state=0)), ('OMP', OrthogonalMatchingPursuit(normalize=False)), ('ARD', ARDRegression())]
                stacking = StackingRegressor(estimators=nnn, final_estimator=ElasticNet(random_state=42, max_iter=int(1e7)), n_jobs=-1)
                stacking.fit(X_train, y_train)
                # es_ = fit_models(new_models, X_train, y_train)
                # stacking_models = {k:v for (k,v) in es_}
                stacking_models = {'stacking': stacking}
        
                # derive_prediction(stacking_models, X_test, str(w) + ' ' + L_list[l], features=X_train.columns.values)
                derive_positions(stacking_models, X_test, y_test, str(w)+' '+L_list[l], X_train.columns.values, new_bee)
                break

        for ind, row in preds_result.iterrows():
            gt = row["gt"]
            nb = row['new_bee']
            # x = range(len(gt))
            # TODO：给定test data 后的预测结果分析
            x = np.arange(600, 1801, 1)
            preds = row["preds"]
            w = row["W_ratio"]
            feature_num = str(len(row["features"]))

            # preds_result.loc[ind, "MSE"] = mean_squared_error(gt, preds)
            # preds_result.loc[ind, "MAE"] = mean_absolute_error(gt, preds)
            # preds_result.loc[ind, "RMSE"] = np.sqrt(mean_squared_error(gt, preds))
            # preds_result.loc[ind, "R2"] = r2_score(gt, preds)

            name = row["model"] + " " + feature_num + "feature"
            eval(x, nb, gt, preds, name, w, save_fig=True)
            # eval_new(Impact_Temp_Rise, preds, name, w, save_fig=True)
            # eval_new(x, preds, name, w, save_fig=True)
            
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






