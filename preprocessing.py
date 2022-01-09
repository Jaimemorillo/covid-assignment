import pandas as pd
from sklearn import preprocessing
from collections import defaultdict

df = pd.read_csv(
    ".//COVID19_data.csv",
    sep=',', 
    header='infer',
    index_col = "ID"
)
# remove DESTIONATION column
df.pop("DESTINATION")

# fetch categorical columns before doing any other preprocessing
cat_mask = (df.dtypes == object)
cat_cols= df.columns[cat_mask].tolist()

# remove all rows containing NA values (47 of them)
df = df.dropna()

# drop all rows where the important values are all 0
df = df.drop(df[(df["TEMP"] == 0) & (df["HEART_RATE"] == 0) &  (df["GLUCOSE"] == 0) & ( df["SAT_O2"] == 0) & (df["BLOOD_PRES_SYS"] == 0) & (df["BLOOD_PRES_DIAS"] == 0)].index)

# only remove the very big outliers, the smaller ones might be relevant
df = df.drop(df[(df["HEART_RATE"] > 400)].index)
df = df.drop(df[(df["BLOOD_PRES_SYS"] > 600)].index)
df = df.drop(df[(df["BLOOD_PRES_DIAS"] > 400)].index)
df = df.drop(df[(df["AGE"] > 180)].index)

# create 2 separate dataframes, categorical and numerical
df_cat = df[cat_cols]
df_num = df.drop(cat_cols, axis = 1)
# does a bit encoding for SEX and EXITUS
d = defaultdict(preprocessing.LabelEncoder)
df_cat_le = df_cat.apply(lambda col: d[col.name].fit_transform(col))

df_preprocessed = pd.merge(left = df_cat_le, right = df_num, on = "ID")
class_col = df_preprocessed.pop("EXITUS")