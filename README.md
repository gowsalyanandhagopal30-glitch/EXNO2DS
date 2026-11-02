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
```
Developed by: N.Gowsalya
Register no: 212225230085
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic_dataset.csv")

df
```
<img width="1310" height="512" alt="Screenshot 2026-11-02 153738" src="https://github.com/user-attachments/assets/50499729-8170-4373-9dad-3345dda6b966" />
```
df.info()
```
<img width="420" height="410" alt="Screenshot 2026-11-02 153834" src="https://github.com/user-attachments/assets/bcbd1ec8-5176-4544-bcbf-118be801d55a" />
```
df.set_index("PassengerId", inplace=True)

df.describe()
```
<img width="675" height="358" alt="Screenshot 2026-11-02 153924" src="https://github.com/user-attachments/assets/1f1c155a-46ba-40eb-a247-e7465bd73f08" />
```
df.shape
```
<img width="412" height="42" alt="Screenshot 2026-11-02 161752" src="https://github.com/user-attachments/assets/0a7130af-a213-4971-a16f-0102129b1dce" />
```
df.nunique()
```
<img width="255" height="263" alt="Screenshot 2026-11-02 154120" src="https://github.com/user-attachments/assets/7f49c7c4-1dd2-46d8-be3f-78c34699748f" />

```
df["Survived"].value_counts()
```
<img width="333" height="97" alt="Screenshot 2026-11-02 155157" src="https://github.com/user-attachments/assets/9d97404a-0034-4da6-875e-2707d0a30e70" />
```
per = (df["Survived"].value_counts() / df.shape[0] * 100).round(2)
per
```
<img width="318" height="92" alt="Screenshot 2026-11-02 155206" src="https://github.com/user-attachments/assets/0d1f64ac-ff7a-4d2b-a108-12ee013032c7" />
```
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.show()
```
<img width="808" height="562" alt="Screenshot 2026-11-02 155413" src="https://github.com/user-attachments/assets/b91a6a73-8554-4953-8b9a-753c83876428" />

```
df
```
<img width="1282" height="561" alt="Screenshot 2026-11-02 155434" src="https://github.com/user-attachments/assets/0e1b2252-784a-4d35-87fc-0efe2be3cf69" />
```
df["Pclass"].unique()
```
<img width="193" height="42" alt="Screenshot 2026-11-02 155444" src="https://github.com/user-attachments/assets/ec6c0357-a6cf-4c95-b2c8-7df95a41985d" />
```
df.rename(columns={"Sex": "Gender"}, inplace=True)
df.head()
```
<img width="1317" height="295" alt="Screenshot 2026-11-02 155502" src="https://github.com/user-attachments/assets/f9e9d405-03f4-4201-a6af-4ffd5dd94a48" />

```
sns.catplot(
    x="Gender",
    col="Survived",
    kind="count",
    data=df,
    height=5,
    aspect=0.7
)
plt.show()
```
<img width="918" height="621" alt="Screenshot 2026-11-02 155521" src="https://github.com/user-attachments/assets/440964d1-96b2-482d-bd9e-4cb680d3e4a9" />
```
df.boxplot(column="Age", by="Survived")

plt.title("Age Distribution Based on Survival")
plt.suptitle("")
plt.show()
```
<img width="798" height="561" alt="Screenshot 2026-11-02 155537" src="https://github.com/user-attachments/assets/cf59ea87-b312-454d-8e4f-c628aed9e1ea" />
```
sns.scatterplot(x=df["Age"], y=df["Fare"])

plt.title("Age vs Fare")
plt.show()
```
<img width="761" height="560" alt="Screenshot 2026-11-02 155554" src="https://github.com/user-attachments/assets/0903ba6c-eae3-4b15-bffe-2b64b52bfbc0" />

```
sns.jointplot(x="Age", y="Fare", data=df)
plt.show()
```
<img width="768" height="735" alt="Screenshot 2026-11-02 155610" src="https://github.com/user-attachments/assets/28ca216a-ca5e-423c-b39f-49378cecaec6" />

```
fig, ax1 = plt.subplots(figsize=(8, 5))
sns.boxplot(
    ax=ax1,
    x="Pclass",
    y="Age",
    hue="Gender",
    data=df
)
plt.title("Age Based on Passenger Class and Gender")
plt.show()
```
<img width="975" height="592" alt="Screenshot 2026-11-02 155627" src="https://github.com/user-attachments/assets/b9fda9a8-5539-49f4-bcad-95720af44224" />

```
sns.catplot(
    data=df,
    col="Survived",
    x="Gender",
    hue="Pclass",
    kind="count"
)

plt.show()

```
<img width="1352" height="613" alt="Screenshot 2026-11-02 155650" src="https://github.com/user-attachments/assets/ffb624ec-1a31-4531-9ba2-6dfa1a5fb478" />

```
corr = df.corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
```
<img width="1022" height="675" alt="Screenshot 2026-11-02 155712" src="https://github.com/user-attachments/assets/011d83bd-c85d-48b2-b295-8fcf8e4ffb32" />

```
sns.pairplot(df)
plt.show()
```
<img width="708" height="685" alt="Screenshot 2026-11-02 155803" src="https://github.com/user-attachments/assets/7f2afe3b-5f2f-4932-92fe-7f248f777441" />

# RESULT
        We have performed Exploratory Data Analysis on the given data set successfully.
