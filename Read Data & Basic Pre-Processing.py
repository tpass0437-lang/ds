#Read Data & Basic Pre-Processing
import pandas as pd
import numpy as np

df = pd.read_csv("C:/dataset/iris-write-from-docker.csv")  
# 🔴 CHANGE FILE PATH to your dataset location

print("First 5 rows")
print(df.head())

print("Last 5 rows")
print(df.tail())

print("Dataset Information")
print(df.info())

print("Statistical Summary")
print(df.describe())

print("Shape of dataset (rows, columns)")
print(df.shape)

print("Column Names")
print(df.columns)

print("Data Types")
print(df.dtypes)

print("Missing Values")
print(df.isnull().sum())

df = df.fillna(df.mean(numeric_only=True))  
# 🔴 If dataset has different numeric columns, check before filling

z = np.abs((df.select_dtypes(include=np.number) -
            df.select_dtypes(include=np.number).mean()) /
           df.select_dtypes(include=np.number).std())

df = df[(z < 3).all(axis=1)]

print("Data after removing outliers:")
print(df.head())

# Filtering, Sorting, Grouping

# 🔴 CHANGE COLUMN NAME inside quotes if needed
filtered = df[df["sepal_length"] > 5]
# 🔴 CHANGE VALUE 5 if required

print("Filtered Data:")
print(filtered.head())

# 🔴 CHANGE COLUMN NAME if required
sorted_df = df.sort_values(by="sepal_width", ascending=False)

print("Sorted Data:")
print(sorted_df.head())

# 🔴 CHANGE GROUP COLUMN NAME if required
grouped = df.groupby("class").mean(numeric_only=True)

print("Grouped Data:")
print(grouped)

