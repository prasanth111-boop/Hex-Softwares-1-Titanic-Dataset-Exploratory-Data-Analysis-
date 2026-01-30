HEX SOFTWARES INTERNSHIP PROJECT 1

# Titanic Data Analysis and Visualization

## Project Overview
This project performs **Exploratory Data Analysis (EDA)** on a Titanic-inspired dataset with 300+ passengers and 18 features. The goal is to:

- Understand dataset structure
- Handle missing values
- Perform statistical analysis
- Engineer useful features
- Visualize patterns and correlations

---

## Dataset Description
The dataset contains the following columns:

- `PassengerId` → Unique passenger ID  
- `Survived` → 0 = No, 1 = Yes  
- `Passenger_Class` → 1, 2, 3  
- `Name` → Passenger Name  
- `Sex` → Male/Female  
- `Age` → Age in years  
- `SibSp` → Siblings/Spouses aboard  
- `Parch` → Parents/Children aboard  
- `Ticket` → Ticket number  
- `Fare` → Ticket fare  
- `Cabin` → Cabin number (missing values)  
- `Embarked` → Port of embarkation (S, C, Q)  
- `Nationality` → Passenger country  
- `Travel_Purpose` → Business/Tourism/Migration  
- `Is_Alone` → 0 = Not Alone, 1 = Alone  
- `Deck` → Deck letter (missing values)  
- `Port_Name` → Port name  
- `Ticket_Type` → First/Second/Third Class

---

## Tools & Libraries
- Python 3  
- Pandas  
- NumPy  
- Matplotlib  
- Seaborn  
- Jupyter Notebook / Google Colab

---

## Key Steps
1. Load dataset  
2. Data cleaning:
   - Handle missing values
   - Drop irrelevant columns
   - Correct data types
3. Feature engineering (`FamilySize`, `Age_Group`, `Is_Adult`, etc.)  
4. Statistical analysis:
   - Summary statistics
   - Grouping, aggregation  
5. Data visualization:
   - Histograms, count plots, box plots, heatmaps, scatter plots  
6. Save cleaned dataset

---

## Insights
- Females had higher survival rate than males  
- Higher-class passengers survived more  
- Children and young adults survived more  
- Fare is positively correlated with survival  
- Most passengers traveled alone or in small families  

---

---

# 2️⃣ Titanic_EDA.ipynb (Google Colab / Jupyter)

```python
# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
from google.colab import files
uploaded = files.upload()  # upload titanic_extended_dataset.csv

df = pd.read_csv("titanic_extended_dataset.csv")
df.head()

# -----------------------------
# 3. DATA INSPECTION
# -----------------------------
df.tail()
df.shape
df.columns
df.info()
df.describe(include='all')
df.nunique()

# -----------------------------
# 4. MISSING VALUES
# -----------------------------
df.isnull().sum()
(df.isnull().sum()/len(df))*100
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop(columns=['Cabin'], inplace=True)

# -----------------------------
# 5. FEATURE ENGINEERING
# -----------------------------
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['Age_Group'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=['Child','Teen','Adult','MiddleAge','Senior'])
df['Is_Adult'] = df['Age'] >= 18
df['Sex'] = df['Sex'].replace({'male':0,'female':1})
df.rename(columns={'Pclass':'Passenger_Class'}, inplace=True)

# -----------------------------
# 6. FILTERING & SORTING
# -----------------------------
df[df['Age']>50]
df.sort_values(by='Fare', ascending=False)
df[['Name','Age','Fare']]
df.drop_duplicates(inplace=True)
df.duplicated().sum()

# -----------------------------
# 7. GROUPING & AGGREGATION
# -----------------------------
df.groupby('Sex')['Survived'].mean()
df.groupby(['Passenger_Class','Sex'])['Survived'].mean()
df.groupby('Passenger_Class').agg({'Fare':'mean','Survived':'mean'})
df.groupby('Embarked')['PassengerId'].count()
pd.pivot_table(df, index='Sex', columns='Passenger_Class', values='Fare', aggfunc='mean')

# -----------------------------
# 8. VISUALIZATIONS
# -----------------------------
# Histogram
sns.histplot(df['Age'], bins=30, kde=True)
plt.show()

# Count plot
sns.countplot(x='Survived', data=df)
plt.show()

# Survival by Gender
sns.countplot(x='Sex', hue='Survived', data=df)
plt.show()

# Box plot of Fare
sns.boxplot(x='Passenger_Class', y='Fare', data=df)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.show()

# Scatter plot
sns.scatterplot(x='Age', y='Fare', hue='Survived', data=df)
plt.show()

# -----------------------------
# 9. SAVE CLEANED DATA
# -----------------------------
df.to_csv("cleaned_titanic_dataset.csv", index=False)

# -----------------------------
# 10. SAMPLE RANDOM ROWS
# -----------------------------
df.sample(10)

