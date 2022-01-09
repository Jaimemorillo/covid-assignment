# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 17:26:46 2022

@author: Víctor Lopo , Aron Latis , Jaime Morillo , Kasper Lange

"""
#%% Imports

import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np

#%% Get data

df = pd.read_csv('COVID19_data.csv', sep=',', header= 0, index_col = 'ID',
                 names = ['ID', 'AGE', 'SEX', 'DAYS_HOSPITAL', 'DAYS_ICU', 'EXITUS', 'DESTINATION', 'TEMP', 'HEART_RATE', 'GLUCOSE', 'SAT_02', 'BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS'])


#%% Sex and Exitus analysis 

number_male_yes = df[df.apply(lambda x: x['SEX'] == 'MALE' and x['EXITUS'] == 'YES', axis=1)]['EXITUS'].count()
number_female_yes = df[df.apply(lambda x: x['SEX'] == 'FEMALE' and x['EXITUS'] == 'YES', axis=1)]['EXITUS'].count()
number_male_no = df[df.apply(lambda x: x['SEX'] == 'MALE' and x['EXITUS'] == 'NO', axis=1)]['EXITUS'].count()
number_female_no = df[df.apply(lambda x: x['SEX'] == 'FEMALE' and x['EXITUS'] == 'NO', axis=1)]['EXITUS'].count()

total_yes = number_female_yes + number_male_yes
total_no = number_female_no + number_male_no

# Female  and Male comparison when exitus = YES

#define data
data = [number_male_yes * 100 / total_yes, number_female_yes * 100 / total_yes]
labels = ['Male', 'Female']

#define Seaborn color palette to use
colors = sns.color_palette('bright')[0:2]

#create pie chart
plt.figure()
plt.title('Male and Female comparison when exitus = YES')
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.show()

#define data
data = [number_male_no * 100 / total_no, number_female_no * 100 / total_no]
labels = ['Male', 'Female']

#define Seaborn color palette to use
colors = sns.color_palette('bright')[0:2]

#create pie chart
plt.figure()
plt.title('Male and Female comparison when exitus = NO')
plt.pie(data, labels = labels, colors = colors, autopct='%.0f%%')
plt.show()

# Let's see now a cross table of the Sex and the Exitus

pd.crosstab(index=df['EXITUS'],
            columns=df['SEX'], margins=True)

# Cross table with percentages 

pd.crosstab(index=df['EXITUS'], columns=df['SEX'],
            margins=True).apply(lambda r: r/len(df) *100,
                                axis=1)
                                
#%% Age vs Exitus analysis

df['AGE_RANK'] = ['R1' if df['AGE'][i]<50 else 'R2' for i in df.index]

number_R1_yes = df[df.apply(lambda x: x['AGE_RANK'] == 'R1' and x['EXITUS'] == 'YES', axis=1)]['EXITUS'].count()
number_R2_yes = df[df.apply(lambda x: x['AGE_RANK'] == 'R2' and x['EXITUS'] == 'YES', axis=1)]['EXITUS'].count()
number_R1_no = df[df.apply(lambda x: x['AGE_RANK'] == 'R1' and x['EXITUS'] == 'NO', axis=1)]['EXITUS'].count()
number_R2_no = df[df.apply(lambda x: x['AGE_RANK'] == 'R2' and x['EXITUS'] == 'NO', axis=1)]['EXITUS'].count()

number_R1 = number_R1_yes + number_R1_no
number_R2 = number_R2_yes + number_R2_no

Classes = ['YES', 'NO']
rank_1 = [number_R1_yes, number_R1_no]
rank_2 = [number_R2_yes, number_R2_no]

X_axis = np.arange(len(Classes))

plt.bar(X_axis - 0.2, rank_1, 0.4, label = 'Rank_1')
plt.bar(X_axis + 0.2, rank_2, 0.4, label = 'Rank_2')
  
plt.xticks(X_axis, Classes)
plt.xlabel("Exitus")
plt.ylabel("Number of patients")
plt.title("Number of patients in each rank and their exitus value")
plt.legend()
plt.show()

#%% Days in hospital and UCI vs Exitus analysis

# Create a new column in the df with the total number of days
df['TOTAL_DAYS'] = [df['DAYS_HOSPITAL'][i] + df['DAYS_ICU'][i] for i in df.index]
df['DAYS_RANK'] = ['R1' if df['TOTAL_DAYS'][i]<20 else 
                   'R2' if (df['TOTAL_DAYS'][i]>20 and df['TOTAL_DAYS'][i]<40) else 
                   'R3' if (df['TOTAL_DAYS'][i]>40 and df['TOTAL_DAYS'][i]<60) else 
                   'R4' if (df['TOTAL_DAYS'][i]>60 and df['TOTAL_DAYS'][i]<80) else
                   'R5' for i in df.index]


plot = pd.crosstab(index=df['DAYS_RANK'],
            columns=df['EXITUS']
                  ).apply(lambda r: r/r.sum() *100,
                          axis=1).plot(kind='bar')

#%% Temperature vs Heart_rate  

plt.figure()
grouped_by_temp = pd.DataFrame(df.groupby('TEMP')['HEART_RATE'].mean())
eje_x = grouped_by_temp.index
eje_y = grouped_by_temp['HEART_RATE']
plt.plot(eje_x, eje_y)
plt.xlim(35,40)
plt.ylabel('Heart Rate Mean')
plt.xlabel('Temperature')
plt.title('Temperature vs Heart Rate')
plt.show()


#%% Temperature vs (blood presure, both sys and dias)

fig, ax = plt.subplots()
grouped_by_temp_sys = pd.DataFrame(df.groupby('TEMP')['BLOOD_PRES_SYS'].mean())
grouped_by_temp_dias = pd.DataFrame(df.groupby('TEMP')['BLOOD_PRES_DIAS'].mean())
eje_x_sys = grouped_by_temp_sys.index
eje_x_dias = grouped_by_temp_dias.index
eje_y_sys = grouped_by_temp_sys['BLOOD_PRES_SYS']
eje_y_dias = grouped_by_temp_dias['BLOOD_PRES_DIAS']
ax.plot(eje_x_sys, eje_y_sys, label='blood_press_sys')
ax.plot(eje_x_dias, eje_y_dias, label='blood_press_dias')
leg = ax.legend()
plt.xlim(35,40)
plt.ylabel('Blood Pressure')
plt.xlabel('Temperature')
plt.title('Temperature vs Blood Pressure')
plt.show()


#%% Heart Rate vs SAT_02

plt.figure()
grouped_by_HeartRate = pd.DataFrame(df.groupby('HEART_RATE')['SAT_02'].mean())
eje_x = grouped_by_HeartRate.index
eje_y = grouped_by_HeartRate['SAT_02']
plt.plot(eje_x, eje_y)
plt.xlim(35,160)
plt.ylabel('O2 saturation')
plt.xlabel('Heart rate')
plt.title('Heart Rate vs SAT_O2')
plt.show()

#%% Temperature vs SAT_02

plt.figure()
grouped_by_HeartRate = pd.DataFrame(df.groupby('TEMP')['SAT_02'].mean())
eje_x = grouped_by_HeartRate.index
eje_y = grouped_by_HeartRate['SAT_02']
plt.plot(eje_x, eje_y)
plt.xlim(35, 40)
plt.ylabel('O2 saturation')
plt.xlabel('Temperature')
plt.title('Temperature vs SAT_O2')
plt.show()

#%% Correlation Matrix

corr_df = df[['TEMP','HEART_RATE','GLUCOSE','SAT_02','BLOOD_PRES_SYS', 'BLOOD_PRES_DIAS', 'DAYS_HOSPITAL', 'DAYS_ICU']].corr()













