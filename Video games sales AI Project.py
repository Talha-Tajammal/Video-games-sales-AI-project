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

# Load dataset
file_path = 'videogamesales/vgsales.csv'
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Basic information about the dataset:")
print(df.head(50))
print(df.shape)
print(df.columns)
print(df.info())
print(df.isnull().sum())

# Drop rows with any null values
df = df.dropna()
print("\nAfter dropping all null values\n")
print(df.isnull().sum())
print(df.describe())

# Plot the number of games launched on each platform
print("\nPlotting the number of games launched on each platform:")
plt.figure(figsize=(10, 6))
sns.countplot(x='Platform', data=df)
plt.xticks(rotation=90)
plt.ylabel('No of Games Launched')
plt.title('Games Launched on Each Platform')
plt.xlabel('Platform Name')
plt.show()

# Plot the number of games launched per year
print("\nPlotting the number of games launched per year:")
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=df)
plt.xticks(rotation=90)
plt.ylabel('No of Games Launched')
plt.xlabel('Year')
plt.title('Games Launched per Year')
plt.show()

# Plot the genre-wise launching of games as a pie chart
print("\nPlotting the genre-wise launching of games:")
df['Genre'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.ylabel("")
plt.title("Genre Wise Launching of Games")
plt.show()

# Display and plot the top 10 publishers by market share
print("\nDisplaying the top 10 publishers by market share:")
print(df['Publisher'].value_counts())
publishers = (df['Publisher'].value_counts() / len(df)) * 100
print(publishers.head(10))
print("\nPlotting the top 10 publishers by market share:")
publishers.head(10).plot(kind='bar')
plt.title("Publisher Wise Launching of Games")
plt.ylabel('Percentage of Market Share')
plt.show()
print("\nCumulative market share of the top 20 publishers:")
print(publishers.head(20).cumsum())

# Print sales totals in different regions
print("\nSales totals in different regions:")
print("North America Sales:", df['NA_Sales'].sum())
print("Europe Sales:", df['EU_Sales'].sum())
print("Japan Sales:", df['JP_Sales'].sum())
print("Other Regions Sales:", df['Other_Sales'].sum())
print("Global Sales:", df['Global_Sales'].sum())

# Plot platform-wise sales per year from 2005 to 2012
print("\nPlotting platform-wise sales per year from 2005 to 2012:")
years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012]
fig, axs = plt.subplots(2, 4, figsize=(15, 7))
axs = axs.flatten()
for i, year in enumerate(years):
    year_data = df[df['Year'] == year]
    year_data['Platform'].value_counts().plot(kind='bar', ax=axs[i], title=str(year))
plt.tight_layout()
plt.show()

# Plot genre-wise sales in North America
print("\nPlotting genre-wise sales in North America:")
genre_sales_na = df.groupby('Genre')['NA_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(15, 7))
plt.bar(genre_sales_na.index, genre_sales_na.values, color='skyblue')
plt.title("Genre-wise Sales in North America")
plt.ylabel("Sales in Million Dollars")
plt.xlabel("Genre")
plt.show()

# Plot genre-wise sales in Europe
print("\nPlotting genre-wise sales in Europe:")
genre_sales_eu = df.groupby('Genre')['EU_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(15, 7))
plt.bar(genre_sales_eu.index, genre_sales_eu.values, color='skyblue')
plt.title("Genre-wise Sales in European Union")
plt.ylabel("Sales in Million Dollars")
plt.xlabel("Genre")
plt.show()

# Plot genre-wise sales in Japan
print("\nPlotting genre-wise sales in Japan:")
genre_sales_jp = df.groupby('Genre')['JP_Sales'].sum().sort_values(ascending=False)
plt.figure(figsize=(13, 6))
plt.bar(genre_sales_jp.index, genre_sales_jp.values, color='skyblue')
plt.title("Genre-wise Sales in Japan")
plt.ylabel("Sales in Million Dollars")
plt.xlabel("Genre")
plt.show()

# Platform-wise sales in different regions (top 10 platforms)
def plot_top_10_sales_by_platform(region):
    platform_sales = df.groupby('Platform')[region].sum().sort_values(ascending=False).head(10)
    return platform_sales

top_10_platform_na = plot_top_10_sales_by_platform('NA_Sales')
top_10_platform_eu = plot_top_10_sales_by_platform('EU_Sales')
top_10_platform_jp = plot_top_10_sales_by_platform('JP_Sales')
top_10_platform_global = plot_top_10_sales_by_platform('Global_Sales')

# Plotting top 10 platform sales in different regions
print("\nPlotting top 10 platform sales in different regions:")
fig, axs = plt.subplots(2, 2, figsize=(15, 7))
platforms = [top_10_platform_na, top_10_platform_eu, top_10_platform_jp, top_10_platform_global]
titles = ['Platform-wise Sales in North America', 'Platform-wise Sales in Europe', 'Platform-wise Sales in Japan', 'Platform-wise Sales (Globally)']
colors = ['green', 'orange', 'skyblue', 'turquoise']
axs = axs.flatten()
for i, sub in enumerate(platforms):
    axs[i].barh(sub.index, sub.values, color=colors[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Total Sales (in millions)')
    axs[i].set_ylabel('Platform')
plt.tight_layout()
plt.show()

# Publisher-wise sales in different regions (top 10 publishers)
def plot_top_10_sales_by_publisher(region):
    publisher_sales = df.groupby('Publisher')[region].sum().sort_values(ascending=False).head(10)
    return publisher_sales

top_10_publishers_na = plot_top_10_sales_by_publisher('NA_Sales')
top_10_publishers_eu = plot_top_10_sales_by_publisher('EU_Sales')
top_10_publishers_jp = plot_top_10_sales_by_publisher('JP_Sales')
top_10_publishers_global = plot_top_10_sales_by_publisher('Global_Sales')

# Plotting top 10 publisher sales in different regions
print("\nPlotting top 10 publisher sales in different regions:")
fig, axs = plt.subplots(2, 2, figsize=(13, 6))
publishers = [top_10_publishers_na, top_10_publishers_eu, top_10_publishers_jp, top_10_publishers_global]
titles = ['Publisher-wise Sales in North America', 'Publisher-wise Sales in Europe', 'Publisher-wise Sales in Japan', 'Publisher-wise Sales (Globally)']
colors = ['green', 'orange', 'skyblue', 'turquoise']
axs = axs.flatten()
for i, sub in enumerate(publishers):
    axs[i].barh(sub.index, sub.values, color=colors[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Total Sales (in millions)')
    axs[i].set_ylabel('Publisher')
plt.tight_layout()
plt.show()

# Year-wise sales in North America (2001 - 2016)
print("\nPlotting year-wise sales in North America (2001 - 2016):")
yearwise_sales_na = df.groupby('Year')['NA_Sales'].sum().loc[2001:2016]
plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_na.index, yearwise_sales_na.values, color='green')
for bar, sales in zip(bars, yearwise_sales_na.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')
plt.title('Yearly Sales in North America (2001 - 2016)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_na.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Year-wise sales in Europe (2001 - 2016)
print("\nPlotting year-wise sales in Europe (2001 - 2016):")
yearwise_sales_eu = df.groupby('Year')['EU_Sales'].sum().loc[2001:2016]
plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_eu.index, yearwise_sales_eu.values, color='orange')
for bar, sales in zip(bars, yearwise_sales_eu.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')
plt.title('Yearly Sales in Europe (2001 - 2016)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_eu.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Year-wise sales in Japan (2001 - 2016)
print("\nPlotting year-wise sales in Japan (2001 - 2016):")
yearwise_sales_jp = df.groupby('Year')['JP_Sales'].sum().loc[2001:2016]
plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_jp.index, yearwise_sales_jp.values, color='skyblue')
for bar, sales in zip(bars, yearwise_sales_jp.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')
plt.title('Yearly Sales in Japan (2001 - 2016)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_jp.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Year-wise sales globally (2001 - 2016)
print("\nPlotting year-wise sales globally (2001 - 2016):")
yearwise_sales_global = df.groupby('Year')['Global_Sales'].sum().loc[2001:2016]
plt.figure(figsize=(12, 6))
bars = plt.bar(yearwise_sales_global.index, yearwise_sales_global.values, color='turquoise')
for bar, sales in zip(bars, yearwise_sales_global.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, f'{sales:.2f}', ha='center', va='bottom')
plt.title('Yearly Sales (Globally: 2001 - 2016)')
plt.xlabel('Year')
plt.ylabel('Total Sales (in millions)')
plt.xticks(yearwise_sales_global.index, rotation=45)
plt.grid(True, axis='y')
plt.tight_layout()
plt.show()

# Top 10 games by sales in different regions
def plot_top_10_sales_by_game(region):
    game_sales = df.groupby('Name')[region].sum().sort_values(ascending=False).head(10)
    return game_sales

top_10_games_na = plot_top_10_sales_by_game('NA_Sales')
top_10_games_eu = plot_top_10_sales_by_game('EU_Sales')
top_10_games_jp = plot_top_10_sales_by_game('JP_Sales')
top_10_games_global = plot_top_10_sales_by_game('Global_Sales')

# Plotting top 10 games by sales in different regions
print("\nPlotting top 10 games by sales in different regions:")
fig, axs = plt.subplots(2, 2, figsize=(20, 10))
Names = [top_10_games_na, top_10_games_eu, top_10_games_jp, top_10_games_global]
titles = ['Top 10 Games Sales in North America', 'Top 10 Games Sales in Europe', 'Top 10 Games Sales in Japan', 'Top 10 Games Sales (Globally)']
colors = ['green', 'orange', 'skyblue', 'turquoise']
axs = axs.flatten()
for i, sub in enumerate(Names):
    axs[i].barh(sub.index, sub.values, color=colors[i])
    axs[i].set_title(titles[i])
    axs[i].set_xlabel('Total Sales (in millions)')
    axs[i].set_ylabel('Games')
plt.tight_layout()
plt.show()
