import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn import preprocessing
from collections import defaultdict
import matplotlib.pyplot as plt

def somewhat_decorator(col_name, plot_type):
    def hist_plot(df):
        df[col_name].plot(
            kind = "hist",
            bins = len(df[col_name].unique())
        )
    def box_plot(df):
        df[col_name].plot(
            kind = "box"    
        )
    if plot_type == "hist":
        return hist_plot
    else:
        return box_plot

df = pd.read_csv(
    "C://Users//User//Desktop//UPM_2021_2022//SEMESTER_1//Data Processes//project//COVID19_data.csv",
    sep=',', 
    header='infer',
    index_col = "ID"
)
cat_mask = (df.dtypes == object)
cat_cols= df.columns[cat_mask].tolist()
df_cat = df[cat_cols]
df_num = df.drop(cat_cols, axis = 1)

d = defaultdict(preprocessing.LabelEncoder)
df_cat_le = df_cat.apply(lambda col: d[col.name].fit_transform(col))

df = pd.merge(left = df_cat_le, right = df_num, on = "ID")

univariate_dict = {}
aux_dict = df.isna().sum()
no_nulls_df = df.dropna()
    
for col_name in df.columns:
    col_dict = {}
    
    col_dict["hist_plot"] = somewhat_decorator(col_name, "hist")
    
    col_dict["box_plot"] = somewhat_decorator(col_name, "box") 

    col_dict["mean"] = no_nulls_df[col_name].mean()
    
    col_dict["std_dev"] = no_nulls_df[col_name].std()
     
    _, p = shapiro(no_nulls_df[col_name])
    col_dict["normality"] = True if (p > 0.5) else False
    
    col_dict["null_count"] = aux_dict[col_name]

    q1, median, q3 = no_nulls_df[col_name].quantile([0.25, 0.5, 0.75])
    iqr = (q3-q1)
    lwr_bound = q1 - 1.5 * iqr    
    upr_bound = q3 + 1.5 * iqr
    col_dict["outlier_indexes"] = no_nulls_df[(no_nulls_df[col_name] > upr_bound) | (no_nulls_df[col_name] < lwr_bound)].index
    col_dict["outliers_count"] = len(col_dict["outlier_indexes"])
    
    col_dict["median"] = median

    univariate_dict[col_name] = col_dict
    

plt.figure()
for index in range(len(univariate_dict.keys())):
    plt.subplot(4, 3, index + 1)
    plt.title(list(univariate_dict.keys())[index])
    univariate_dict[list(univariate_dict.keys())[index]]["hist_plot"](no_nulls_df)
    
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    
plt.figure()
for index in range(len(univariate_dict.keys())):
    plt.subplot(4, 3, index + 1)
    plt.title(list(univariate_dict.keys())[index])
    univariate_dict[list(univariate_dict.keys())[index]]["box_plot"](no_nulls_df)
    
plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)

mean_array, std_array, median_array = [], [], []
plt.figure()
index = 1
for column in univariate_dict:
    mean_array.append(univariate_dict[column]["mean"])
    std_array.append(univariate_dict[column]["std_dev"])
    median_array.append(univariate_dict[column]["median"])
    index += 1
    
plt.bar(np.arange(1,12) - 0.2, mean_array, 0.2, label = 'Mean')
plt.bar(np.arange(1,12), std_array, 0.2, label = 'Std Dev')
plt.bar(np.arange(1,12) + 0.2, median_array, 0.2, label = 'Median')
plt.xticks(np.arange(1,12), df.columns, rotation = 45)
plt.xlabel("Variables")
plt.ylabel("Numeric Values")
plt.legend()
plt.show()