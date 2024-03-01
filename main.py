import numpy as np
import pandas as pd
import seaborn as sns
# import matplotlib.pyplot as plt
# from scipy.interpolate import interp1d
import sklearn.model_selection as ms
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor as XGBR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error #MSE
from sklearn.metrics import mean_absolute_error #MAE
from sklearn.metrics import r2_score#R 2
from eval import eval

# translate all data to .csv
def xlsx_to_csv_pd():
    data_xls = pd.read_excel("C:/Users/xiaol/PycharmProjects/zzw/Data/data.xlsx", index_col=0)
    data_xls.to_csv('C:/Users/xiaol/PycharmProjects/zzw/Data/data.csv', encoding='utf-8')

# normalize function z_score
def normalize_zscore(column):
    return (column - column.mean()) / column.std()


# normalize function min-max
def normalize_minmax(column):
    return (column - column.min()) / column.max()-column.min()

def train_test_split(df, data_norm, num):
    # split according to rate of W
    test_data = pd.concat([data_norm.iloc[Ti_group[num][0]:Ti_group[num][1]], data_norm.iloc[Zr_group[num][0]:Zr_group[num][1]]])
    x_axis = pd.concat([df.iloc[Ti_group[num][0]:Ti_group[num][1]], df.iloc[Zr_group[num][0]:Zr_group[num][1]]])
    select_test_indices = list(range(Ti_group[num][0], Ti_group[num][1])) + list(range(Zr_group[num][0], Zr_group[num][1]))
    train_data = data_norm.drop(select_test_indices)
    return train_data, test_data, x_axis

rf_params = [{
    'max_depth': [5, 10,15],
    'n_estimators': [10, 20, 30],
    'max_features': ["sqrt", "log2"],
}]

xgb_params = [{
    'max_depth': [9],  # 2-15
    'min_child_weight':[2],
    'booster':['gbtree', 'dart'],
    'eta':[0.2, 0.3, 0.4]
}]

svr_params = [
    {'kernel': ['sigmoid'], 'C': [1, 10, 100, 1000]},
]

dt_params =[{
    'max_depth':[5, 10, 15],
    'splitter':['best', 'random'],
}]

gbr_params = [{
    'n_estimators': [10, 20, 30],
    'max_depth': [5, 10, 15],
    'min_samples_split': [5,],
    'learning_rate': [0.01],
    'loss': ['squared_error'],
}]

params = {
    'ridge__alpha': [0.1, 1.0, 10.0]
    }

models = {
    'LR': LinearRegression(),
    'DT':ms.GridSearchCV(DecisionTreeRegressor(random_state=1), dt_params, cv=3),
    # 'RF': ms.GridSearchCV(RandomForestRegressor(random_state=1), rf_params, cv=3),
    # 'GBR': ms.GridSearchCV(GradientBoostingRegressor(random_state=1), gbr_params, cv=3),
    # 'XGBR': ms.GridSearchCV(XGBR(),xgb_params, cv=3),
    'SVR':ms.GridSearchCV(SVR(),svr_params, cv=3),
}

estimators = [
    ('LR', LinearRegression()),
    ('ridge', RidgeCV()),
    # ('lasso', LassoCV(random_state=42)),
    # ('knr', KNeighborsRegressor(n_neighbors=20,
    #                             metric='euclidean')),
    ('svr_lin', SVR(kernel='linear', gamma='auto')),
    ('svr_rbf', SVR(kernel='rbf', gamma='auto')),
    ('DT', ms.GridSearchCV(DecisionTreeRegressor(random_state=1), dt_params, cv=3)),
    # ('RF', ms.GridSearchCV(RandomForestRegressor(random_state=1), rf_params, cv=3)),
    # ('SVR', ms.GridSearchCV(SVR(), svr_params, cv=3))
]
final_layer_rfr = ms.GridSearchCV(RandomForestRegressor(random_state=1), rf_params, cv=3)
final_layer_xgb = ms.GridSearchCV(GradientBoostingRegressor(random_state=1), gbr_params, cv=3)

final_layer = StackingRegressor(
    estimators=[
        ('rf', final_layer_rfr),
        ('xgb', final_layer_xgb)
    ]
)

def fit_models(x_train,y_train):
    mfit = {model: models[model].fit(x_train, y_train) for model in models.keys()}
    # b_params = {model: models[model].best_params_ for model in models.keys()}
    # print(b_params)
    # b_score = {model: models[model].best_score_ for model in models.keys()}
    # print(b_score)

def derive_positions(x_test):
    for model in models.keys():
        answer['pre'+model] = models[model].predict(x_test)

def get_answer():
    effect = pd.DataFrame(columns = ['model','MSE','MAE','RMSE','R2'])
    i = 0
    for k,v in models.items():
        y_pred = answer['pre'+k]
        MSE = mean_squared_error(y_test,y_pred)
        MAE = mean_absolute_error(y_test,y_pred)
        RMSE = np.sqrt(mean_squared_error(y_test,y_pred))
        R2 = r2_score(y_test,y_pred)
        effect.loc[i] = [k,MSE,MAE,RMSE,R2]
        i += 1
        print(f"MSE:{MSE}MAE:{MAE}RMSE:{RMSE}R2:{R2}")
    print(effect)

def array_to_string(arr):
    return ', '.join(map(str, arr))

if __name__ == "__main__":

    # translate all data to .csv
    # xlsx_to_csv_pd()

    # load all data
    df = pd.read_csv("C:/Users/xiaol/PycharmProjects/zzw/Data/data.csv")

    # remove some features
    # 'Adensity','AC_material_weight','TD_weight','Measure_Velocity','Overpressure'
    df = df.drop(columns=['Adensity','AC_material_weight','TD_weight','Measure_Velocity','Overpressure'])

    # normalize X
    data_norm = df.iloc[:, :-1].apply(normalize_zscore)
    data_norm['React_Efficiency'] = df['React_Efficiency']

    # feature selection
    X = data_norm.iloc[:, :-1].values
    y = data_norm['React_Efficiency'].values
    mi = mutual_info_regression(X, y)
    selector = SelectKBest(mutual_info_regression, k=5)
    X_new = selector.fit_transform(X, y)
    selected_features_idx = selector.get_support(indices=True)

    # new data_norm after selected features
    data_norm = data_norm.iloc[:, selected_features_idx]
    data_norm['React_Efficiency'] = df['React_Efficiency']

    # partition train_data and test_data according to the weight of w: [0, 10, 25, 50, 75]
    Ti_group = {0: [0, 6], 10: [6, 12], 25: [12, 19], 50: [19, 27], 75: [27, 34]}
    Zr_group = {0: [34, 40], 10: [40, 47], 25: [47, 52], 50: [52, 59], 75: [59, 65]}
    W_list = [0, 10, 25, 50, 75]
    preds_result = pd.DataFrame(columns=['model', 'W_ratio', 'x', 'gt', 'preds'])

    i = 0
    for w in W_list:
        train_data, test_data, x_axis = train_test_split(df, data_norm, w)
        X_train = train_data.drop(columns=['React_Efficiency'])
        # X_train = train_data.to_numpy()
        y_train = train_data['React_Efficiency']
        # y_train = y_train.to_numpy()

        X_test = test_data.drop(columns=['React_Efficiency'])
        # X_test = X_test.to_numpy()

        # X_tmp = np.arange(300, 2000, 1)
        # X_test = normalize_zscore(X_tmp).reshape(-1, 1)

        y_test = test_data['React_Efficiency']
        # y_test = y_test.to_numpy()
        # x_axis = x_axis['Impact_Temp_Rise']     # no normalized data
        # x_axis = x_axis.to_numpy()
        answer = pd.DataFrame()
        fit_models(X_train, y_train)
        derive_positions(X_test)
        get_answer()
        for k, v in models.items():
            gt = y_test
            x = range(len(gt))
            preds = answer['pre' + k]
            eval(x, gt, preds, k, w, save_fig=False)
            # preds_result.loc[i] = [k, w, x_axis, np.array(gt), np.array(preds)]
            i += 1

        # predict by ensemble model
        sclf = StackingRegressor(
            estimators=estimators,
            final_estimator=final_layer
        )

        sclf.fit(X_train, y_train)
        en_pred = sclf.predict(X_test)
        eval(x, gt, en_pred, 'Ensemble', w, save_fig=False)

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






