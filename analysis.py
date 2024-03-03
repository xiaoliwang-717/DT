"""
2024-3-3
----
1. install requirements.txt
2. See TODO.
"""
import pandas as pd
import numpy as np
from omnixai.data.tabular import Tabular
import sklearn
from sklearn.linear_model import Lasso
from omnixai.preprocessing.tabular import TabularTransform
from omnixai.explainers.tabular import TabularExplainer
from omnixai.visualization.dashboard import Dashboard
from omnixai.explainers.prediction import PredictionAnalyzer


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

df = pd.read_csv("./data.csv")
# print(df)
# remove some features
# 'Adensity','AC_material_weight','TD_weight','Measure_Velocity','Overpressure'
df = df.drop(columns=['Adensity','AC_material_weight','TD_weight','Measure_Velocity','Overpressure'])

# normalize X
data_norm = df.iloc[:, :-1].apply(normalize_zscore)
data_norm['React_Efficiency'] = df['React_Efficiency']

# partition train_data and test_data according to the weight of w: [0, 10, 25, 50, 75]
Ti_group = {0: [0, 6], 10: [6, 12], 25: [12, 19], 50: [19, 27], 75: [27, 34]}
Zr_group = {0: [34, 40], 10: [40, 47], 25: [47, 52], 50: [52, 59], 75: [59, 65]}
W_list = [0, 10, 25, 50, 75]
preds_result = pd.DataFrame(columns=['model', 'W_ratio', 'x', 'gt', 'preds'])

# TODO: 这里是将我们的数据转换为OMNIXAI的表格类。很简单，把我们预测的那一项放在target_columns就行
tabular_data = Tabular(
    df,
    target_column='React_Efficiency'
)

# TODO: 这里一般是预处理。不用怎么变动。
# Data preprocessing
transformer = TabularTransform().fit(tabular_data)
class_names = transformer.class_names
# TODO: 这里是将表格数据转换成训练矩阵的形式。
x = transformer.transform(tabular_data)
# Split into training and test datasets
train, test, train_labels, test_labels = \
    sklearn.model_selection.train_test_split(x[:, :-1], x[:, -1], train_size=0.80)
# TODO: 训练好的模型放在这里。可以接一个接口进来。
model = Lasso()
model.fit(train, train_labels)

# 这里是将训练数据转换称为解释库的表格数据，这一步很重要，为了分析用。
# Convert the transformed data back to Tabular instances
train_data = transformer.invert(train)
test_data = transformer.invert(test)



# Initialize a TabularExplainer
explainer = TabularExplainer(
   # TODO: 这里是解释器，需要去官网看看还有那些解释器可以用
   explainers=["lime", "shap", "ale"], # The explainers to apply
   mode="regression",                             # The task type
   data=train_data,                                   # The data for initializing the explainers
   model=model,                                       # The ML model to explain
   preprocess=lambda z: transformer.transform(z),     # Converts raw features into the model inputs
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
    test_data=test_data,                           # The test dataset (a `Tabular` instance)
    test_targets=test_labels,                      # The test labels (a numpy array)
    model=model,                                   # The ML model
    preprocess=lambda z: transformer.transform(z)  # Converts raw features into the model inputs
)
prediction_explanations = analyzer.explain()


# Launch a dashboard for visualization
dashboard = Dashboard(
    instances=test_instances,                        # The instances to explain
    local_explanations=local_explanations,           # Set the generated local explanations
    global_explanations=global_explanations,         # Set the generated global explanations
    prediction_explanations=prediction_explanations, # Set the prediction metrics
    class_names=class_names,                         # Set class names
    explainer=explainer                              # The created TabularExplainer for what if analysis
)
dashboard.show(host='127.0.0.1', port='8888')                                     # Launch the dashboard