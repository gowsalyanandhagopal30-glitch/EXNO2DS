# EXNO2DS
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT

Developed by: Rakisha R
Register no: 212225230223
```
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns  
df=pd.read_csv("titanic_dataset.csv") 
df
```
<img width="920" height="557" alt="image" src="https://github.com/user-attachments/assets/be60ffc5-4db8-4a18-b843-a8ddf63d724d" />
```
df.info()
```
<img width="852" height="377" alt="image" src="https://github.com/user-attachments/assets/75c7f7b7-214e-456f-8633-940585fc823f" />

```
df.set_index("PassengerId",inplace=True) 
df.describe()
```
<img width="902" height="321" alt="image" src="https://github.com/user-attachments/assets/066d3664-c752-4f47-90db-22e9775b03e9" />
```
df.shape
```
(891, 11)
```
df.nunique()
```
<img width="877" height="295" alt="image" src="https://github.com/user-attachments/assets/0b922171-0705-4ce0-b284-893b65433f67" />
```
df["Survived"].value_counts()
```
<img width="770" height="97" alt="image" src="https://github.com/user-attachments/assets/13994618-8b51-4ce0-be9b-a37384799e16" />

```
per=(df["Survived"].value_counts()/df.shape[0]*100).round(2) 
per
```
<img width="926" height="97" alt="image" src="https://github.com/user-attachments/assets/3bf384e3-b626-465e-b2d4-9d642e232295" />

```
sns.countplot(data=df,x="Survived")
```
<img width="867" height="627" alt="image" src="https://github.com/user-attachments/assets/fc489216-a227-45fa-92cc-c8ba0759f988" />
```
df
```
<img width="882" height="497" alt="image" src="https://github.com/user-attachments/assets/7ce51c93-0193-4def-9722-a200a3d384bc" />
```
df.Pclass.unique()
```
<img width="770" height="42" alt="image" src="https://github.com/user-attachments/assets/5e6feba7-24af-4fdf-ba19-9e6520e4eddc" />

```
df.rename(columns={'Sex':'Gender'},inplace=True) 
df
```
<img width="861" height="485" alt="image" src="https://github.com/user-attachments/assets/fb1b900f-4ed6-4579-a271-8cb3775d0d5d" />
```
sns.catplot(x="Gender",col="Survived",kind="count",data=df,height=5,aspect=.7)
```

<img width="815" height="547" alt="image" src="https://github.com/user-attachments/assets/0b7e83f0-29ad-4ea8-8641-cc9a7542da7e" />

```
sns.catplot(x="Survived",hue="Gender",data=df,kind="count")
```
<img width="825" height="546" alt="image" src="https://github.com/user-attachments/assets/be1f2336-4345-4163-b7e7-accf3c713385" />

```
df.boxplot(column="Age",by="Survived")
```
<img width="812" height="501" alt="image" src="https://github.com/user-attachments/assets/49b16f1d-f9e9-4c8d-a09b-3dda75eda58f" />
```
sns.scatterplot(x=df["Age"],y=df["Fare"])
```
<img width="805" height="496" alt="image" src="https://github.com/user-attachments/assets/6adafec7-1ab5-46ed-8802-4be4e59f6e89" />
```
sns.jointplot(x="Age",y="Fare",data=df)
```
 <img width="822" height="647" alt="image" src="https://github.com/user-attachments/assets/fd9fab3d-6bc7-44b9-bf69-adcaac742c7a" />

```
fig, ax1 = plt.subplots(figsize=(8,5)) 
plt = sns.boxplot(ax=ax1,x='Pclass',y='Age',hue='Gender',data=df)
```
<img width="857" height="475" alt="image" src="https://github.com/user-attachments/assets/bba6a2a3-81f9-4c71-af1a-c72b80c3dcf5" />

```
sns.catplot(data=df,col="Survived",x="Gender",hue="Pclass",kind="count")
```
<img width="831" height="432" alt="image" src="https://github.com/user-attachments/assets/0493f267-63d6-4b3d-bc1d-60e9381abe03" />
```
corr=df.corr() 
sns.heatmap(corr,annot=True)
```
<img width="842" height="475" alt="image" src="https://github.com/user-attachments/assets/ec6388fa-0ca5-4767-b3a3-d0f671cca7ff" />

```
sns.pairplot(df)
```
<img width="902" height="820" alt="image" src="https://github.com/user-attachments/assets/330c3ccb-66c9-4e38-a8bc-b50b23ebb43f" />

# RESULT
        We have performed Exploratory Data Analysis on the given data set successfully.
