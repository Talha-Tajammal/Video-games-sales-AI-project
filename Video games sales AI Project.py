# Commands for upload projet on git.
# 
# cd path/to/your/project
# git init
# git remote add origin https://github.com/YourUsername/YourRepositoryName.git
# git add .
# git commit -m "Initial commit"
# git push origin master  # or 'master' if that's your default branch




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

file_path = 'videogamesales/vgsales.csv'
df = pd.read_csv(file_path)

# print(df.head(50))


# print(df.shape)

# print(df.columns)

# print(df.info())

# print(df.isnull().sum())

# df=df.dropna()
# print("After drop all null values \n " )
# print(df.isnull().sum())

# print(df.describe())


# plt.figure(figsize=(10,6))
# sns.countplot(x='Platform',data=df)
# plt.xticks(rotation=90)
# plt.ylabel('No of Games Launched')
# plt.title('Games Launched on Each Platform')
# plt.xlabel('Platform Name')
# plt.show()

# plt.figure(figsize=(10,6))
# sns.countplot(x='Year',data=df)
# plt.xticks(rotation=90)
# plt.ylabel('No of Games Launched')
# plt.xlabel('Year')
# plt.title('Games Launched per Year')
# plt.show()


# df['Genre'].value_counts().plot(kind='pie',autopct='%1.1f%%')
# plt.ylabel("")
# plt.title("Genre Wise Launching of Games")
# plt.show()

# print(df['Publisher'].value_counts())

# publishers = (df['Publisher'].value_counts() / len(df)) * 100
# print(publishers.head(10) )#Top 10 publishers

# publishers.head(10).plot(kind='bar')
# plt.title("Publisher Wise Launching of Games")
# plt.ylabel('Percentage of Market Share')
# plt.show()

# print(publishers.head(20).cumsum())

# print(df['NA_Sales'].sum())
# print(df['EU_Sales'].sum())
# print(df['JP_Sales'].sum())
# print(df['Other_Sales'].sum())
# print(df['Global_Sales'].sum())


# year_2005 = df[df['Year']==2005.0]
# year_2006 = df[df['Year']==2006.0]
# year_2007 = df[df['Year']==2007.0]
# year_2008 = df[df['Year']==2008.0]
# year_2009 = df[df['Year']==2009.0]
# year_2010 = df[df['Year']==2010.0]
# year_2011 = df[df['Year']==2011.0]
# year_2012 = df[df['Year']==2012.0]

# fig, axs = plt.subplots(2, 4, figsize=(15, 7))
# years = [year_2005, year_2006, year_2007, year_2008, year_2009, year_2010, year_2011,year_2012]
# titles = ['2005', '2006', '2007', '2008', '2009', '2010', '2011','2012']
# axs = axs.flatten()
# for i in range(len(years)):
#     years[i]['Platform'].value_counts().plot(kind='bar', ax=axs[i], title=titles[i])
# plt.tight_layout()
# plt.show()


# genre_sales = df.groupby('Genre')['NA_Sales'].sum().reset_index()
# print(genre_sales)
# genre_sales = genre_sales.sort_values(by='NA_Sales', ascending=False)
# plt.figure(figsize=(15, 7))
# plt.bar(genre_sales['Genre'], genre_sales['NA_Sales'], color='skyblue')
# plt.title("Genre-wise Sales in North America")
# plt.ylabel("Sales in Million Dollars")
# plt.xlabel("Genre")
# plt.show()

# genre_sales_eu = df.groupby('Genre')['EU_Sales'].sum().reset_index()
# print(genre_sales_eu)
# genre_sales_eu = genre_sales_eu.sort_values(by='EU_Sales', ascending=False)
# plt.figure(figsize=(15, 7))
# plt.bar(genre_sales_eu['Genre'], genre_sales_eu['EU_Sales'], color='skyblue')
# plt.title("Genre-wise Sales in European Union")
# plt.ylabel("Sales in Million Dollars")
# plt.xlabel("Genre")
# plt.show()

# genre_sales_jp = df.groupby('Genre')['JP_Sales'].sum().reset_index()
# print(genre_sales_jp)
# genre_sales_jp = genre_sales_jp.sort_values(by='JP_Sales', ascending=False)
# plt.figure(figsize=(13, 6))
# plt.bar(genre_sales_jp['Genre'], genre_sales_jp['JP_Sales'], color='skyblue')
# plt.title("Genre-wise Sales in Japan")
# plt.ylabel("Sales in Million Dollars")
# plt.xlabel("Genre")
# plt.show()

# ## Platform wise Sales in different Regions.

# platform_sales_na = df.groupby('Platform')['NA_Sales'].sum().reset_index()
# top_10_platform_na = platform_sales_na.sort_values(by='NA_Sales', ascending=False).head(10)
# print(top_10_platform_na)

# platform_sales_eu = df.groupby('Platform')['EU_Sales'].sum().reset_index()
# top_10_platform_eu = platform_sales_eu.sort_values(by='EU_Sales', ascending=False).head(10)
# print(top_10_platform_eu)

# platform_sales_jp = df.groupby('Platform')['JP_Sales'].sum().reset_index()
# top_10_platform_jp = platform_sales_jp.sort_values(by='JP_Sales', ascending=False).head(10)
# print(top_10_platform_jp)

# platform_sales_global = df.groupby('Platform')['Global_Sales'].sum().reset_index()
# top_10_platform_global = platform_sales_global.sort_values(by='Global_Sales', ascending=False).head(10)
# print(top_10_platform_global)

# fig, axs = plt.subplots(2, 2, figsize=(15, 7))
# platforms = [top_10_platform_na, top_10_platform_eu, top_10_platform_jp, top_10_platform_global]
# titles = ['Platform-wise Sales in North America', 'Platform-wise Sales in Europe', 'Platform-wise Sales in Japan', 'Platform-wise Sales (Globally)']
# colors = ['green','orange','skyblue','turquoise']
# axs = axs.flatten()
# for i, sub in enumerate(platforms):
#     axs[i].barh(sub.iloc[:,0], sub.iloc[:,1], color=colors[i])
#     axs[i].set_title(titles[i])
#     axs[i].set_xlabel('Total Sales (in millions)')
#     axs[i].set_ylabel('Platform')
# plt.tight_layout()
# plt.show()

# ## publisher wise sales in different reagion

publisher_sales_na = df.groupby('Publisher')['NA_Sales'].sum().reset_index()
top_10_publishers_na = publisher_sales_na.sort_values(by='NA_Sales', ascending=False).head(10)
print(top_10_publishers_na)
publisher_sales_eu = df.groupby('Publisher')['EU_Sales'].sum().reset_index()
top_10_publishers_eu = publisher_sales_eu.sort_values(by='EU_Sales', ascending=False).head(10)
print(top_10_publishers_eu)
publisher_sales_jp = df.groupby('Publisher')['JP_Sales'].sum().reset_index()
top_10_publishers_jp = publisher_sales_jp.sort_values(by='JP_Sales', ascending=False).head(10)
print(top_10_publishers_jp)
publisher_sales_global = df.groupby('Publisher')['Global_Sales'].sum().reset_index()
top_10_publishers_global = publisher_sales_global.sort_values(by='Global_Sales', ascending=False).head(10)
print(top_10_publishers_global)

fig, axs = plt.subplots(2, 2, figsize=(13, 6))
publishers = [top_10_publishers_na, top_10_publishers_eu, top_10_publishers_jp, top_10_publishers_global]
titles = ['Publisher-wise Sales in North America', 'Publisher-wise Sales in Europe', 'Publisher-wise Sales in Japan', 'Publisher-wise Sales (Globally)']
colors = ['green','orange','skyblue','turquoise']
axs = axs.flatten()
for i, sub in enumerate(publishers):
    axs[i].barh(sub.iloc[:,0], sub.iloc[:,1], color=colors[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Total Sales (in millions)')
    axs[i].set_ylabel('Plublisher')
plt.tight_layout()
plt.show()


yearwise_sales_na = df.groupby('Year')['NA_Sales'].sum().sort_values(ascending=False)
yearwise_sales_na = yearwise_sales_na[(yearwise_sales_na.index >= 2001) & (yearwise_sales_na.index <= 2016)]

plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_na.index, yearwise_sales_na.values, color='green')
# Add sales value as text on each bar
for bar, sales in zip(bars, yearwise_sales_na.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')
plt.title('Yearly Sales in North America (2001 - 2020)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_na.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


yearwise_sales_eu = df.groupby('Year')['EU_Sales'].sum().sort_values(ascending=False)
yearwise_sales_eu = yearwise_sales_eu[(yearwise_sales_eu.index >= 2001) & (yearwise_sales_eu.index <= 2016)]

plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_eu.index, yearwise_sales_eu.values, color='orange')

# Add sales value as text on each bar
for bar, sales in zip(bars, yearwise_sales_eu.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')

plt.title('Yearly Sales in Europe (2001 - 2016)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_eu.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()



yearwise_sales_jp = df.groupby('Year')['JP_Sales'].sum().sort_values(ascending=False)
yearwise_sales_jp = yearwise_sales_jp[(yearwise_sales_jp.index >= 2001) & (yearwise_sales_jp.index <= 2016)]

plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_jp.index, yearwise_sales_jp.values, color='skyblue')

# Add sales value as text on each bar
for bar, sales in zip(bars, yearwise_sales_jp.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')

plt.title('Yearly Sales in Japan (2001 - 2020)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_jp.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


yearwise_sales_global = df.groupby('Year')['Global_Sales'].sum().sort_values(ascending=False)
yearwise_sales_global = yearwise_sales_global[(yearwise_sales_global.index >= 2001) & (yearwise_sales_global.index <= 2016)]

plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_global.index, yearwise_sales_global.values, color='turquoise')

# Add sales value as text on each bar
for bar, sales in zip(bars, yearwise_sales_global.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')

plt.title('Yearly Sales (Globally: 2001 - 2020)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_global.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()


















