import pandas as pd

# partition train_data and test_data according to the weight of w: [0, 10, 25, 50, 75]
Ti_group = {0: [0, 6], 10: [6, 12], 25: [12, 19], 50: [19, 27], 75: [27, 34]}
Zr_group = {0: [34, 40], 10: [40, 47], 25: [47, 52], 50: [52, 59], 75: [59, 65]}
def train_test_split(df, data_norm, num):
    # split according to rate of W
    test_data = pd.concat(
        [data_norm.iloc[Ti_group[num][0]:Ti_group[num][1]], data_norm.iloc[Zr_group[num][0]:Zr_group[num][1]]])
    x_axis = pd.concat([df.iloc[Ti_group[num][0]:Ti_group[num][1]], df.iloc[Zr_group[num][0]:Zr_group[num][1]]])
    select_test_indices = list(range(Ti_group[num][0], Ti_group[num][1])) + list(
        range(Zr_group[num][0], Zr_group[num][1]))
    train_data = data_norm.drop(select_test_indices)
    X_train = train_data.drop(columns=['React_Efficiency'])
    # X_train = train_data.to_numpy()
    y_train = train_data['React_Efficiency']
    # y_train = y_train.to_numpy()

    X_test = test_data.drop(columns=['React_Efficiency'])
    # X_test = X_test.to_numpy()

    # X_tmp = np.arange(300, 2000, 1)
    # X_test = normalize_zscore(X_tmp).reshape(-1, 1)

    y_test = test_data['React_Efficiency']
    return X_train, y_train, X_test, y_test


