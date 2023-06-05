
# EXP-10-DATA-SCIENCE
# AIM:
You are given land.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) Plotting different types plot of data visualization using matplotlib.
# Explanation
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.
# ALGORITHM
### STEP 1
Read the given Data

### STEP 2
Get the information about the data

### STEP 3
Detect the Outliers using IQR method and Z score

### STEP 4
Remove the outliers

### STEP 5
Using matplotlib create various plot for visualization
# CODE
``` PYTHON
import pandas as ps
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=ps.read_csv("/content/land.csv")
df

df.head()

df.tail(5)

df.info()

df.isnull().sum()

q1=df['price_per_sqft'].quantile(0.35)
q3=df['price_per_sqft'].quantile(0.65)
print("First Quantile =",q1,"Second quantile =",q3)

from scipy import stats
z=np.abs(stats.zscore(df['price_per_sqft']))
df2=df[(z<3)]
df2

plt.figure(figsize=(12,10))
cols = ['bhk','bath','size']
Q1 = df[cols].quantile(0.25)
Q3 = df[cols].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
plt.title("Dataset after removing outliers")
df.boxplot()
plt.show()

sns.boxplot(x="price_per_sqft",data=df)
plt.figure(figsize=(9,6))

sns.lineplot(x="price",y="bhk",data=df,marker='o')
plt.xticks(rotation = 90)

sns.lineplot(x='price',y='total_sqft', hue ="price",data=df)

sns.scatterplot(x='total_sqft',y='price_per_sqft',data=df)

sns.boxplot(x="price",y="total_sqft",data=df)

sns.barplot(x="bath",y="bhk",data=df)
plt.xticks(rotation = 90)

df3=df.groupby(by=["bath"]).sum()
labels=[]
for i in df3.index:
    labels.append(i) 
plt.figure(figsize=(8,8))
colors = sns.color_palette('pastel')
plt.pie(df3["total_sqft"],colors = colors,labels=labels, autopct = '%0.0f%%')
plt.show()

sns.pointplot(x=df["price_per_sqft"],y=df["bath"])
df.corr()
plt.subplots(figsize=(12,7))
sns.heatmap(df.corr(),annot=True)

```

# OUTPUT
##### READ
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/040974a3-678f-412f-8c67-13da2939bd8f)
##### HEAD
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/3ea36ab2-8c03-4b3c-b018-ac7664411417)
##### TAIL
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/200e9d5d-fa99-4f23-aa65-5431da363305)
#### INFO
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/28dbadef-ffb3-4358-a1d6-3f9a75823191)
#### ISNULL SUM
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/c72b762b-7638-4df0-ab27-86bdd31f11c4)
#### QUANTILE
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/1936022f-6898-4529-9df0-761a5746ca18)
#### DATASET AFTER REMOVAL OF OUTLIER USING Z-SCORE
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/60cbcc85-37f9-4493-9751-e50234c2f1b7)
#### BOX 
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/69e6411c-ecac-4883-927c-22e9ba3e6c74)
#### LINE PLOT 1
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/f58ebafe-0e41-47a4-8214-9e281fb63667)
#### LINE PLOT 2
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/bae6cdca-5762-4ee9-8489-dcc7e2b41242)
#### SCATTER
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/89bb8c01-3799-457f-990b-2ff41ff1f074)
#### BOX
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/c0d12656-9870-48e0-bc9c-ff4a48b078f9)
#### BAR
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/8313143e-da08-47f6-9f2a-334ae723c902)
#### PIE CHAT
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/80e2adfa-fca8-466f-9320-64d0e635f52c)
#### POINT PLOT
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/61f3cff4-8553-4cf2-9ff7-0b0fde07c03a)
#### HEAT MAP
![image](https://github.com/MukeshVelmurugan/EXP-10-DATA-SCIENCE/assets/118707363/be7b1dbb-cde8-4df6-803c-aa5d80053705)

# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods.
