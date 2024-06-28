# Commands for upload projet on git.
# 
# cd path/to/your/project
# git init
# git remote add origin https://github.com/YourUsername/YourRepositoryName.git
# git add .
# git commit -m "Initial commit"
# git push origin master  # or 'master' if that's your default branch




import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'videogamesales/vgsales.csv'
df = pd.read_csv(file_path)

# Drop rows with any null values
df = df.dropna()

# Basic information about the dataset
print("Basic information about the dataset:")
print(df.head(10))
print(df.shape)
print(df.info())
print(df.isnull().sum())

# Plot the number of games launched on each platform
plt.figure(figsize=(10, 6))
sns.countplot(x='Platform', data=df)
plt.xticks(rotation=90)
plt.ylabel('No of Games Launched')
plt.title('Games Launched on Each Platform')
plt.xlabel('Platform Name')
plt.show()

# Plot the number of games launched per year
plt.figure(figsize=(10, 6))
sns.countplot(x='Year', data=df)
plt.xticks(rotation=90)
plt.ylabel('No of Games Launched')
plt.xlabel('Year')
plt.title('Games Launched per Year')
plt.show()

# Plot the genre-wise launching of games as a pie chart
df['Genre'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(8, 8))
plt.title("Genre Wise Launching of Games")
plt.show()

# Top 10 publishers by market share
publishers = (df['Publisher'].value_counts() / len(df)) * 100
print("Top 10 publishers by market share:")
print(publishers.head(10))

# Plot the top 10 publishers by market share
publishers.head(10).plot(kind='bar', figsize=(10, 6))
plt.title("Top 10 Publishers by Market Share")
plt.ylabel('Percentage of Market Share')
plt.show()

# Sales totals in different regions
print("\nSales totals in different regions:")
print("North America Sales:", df['NA_Sales'].sum())
print("Europe Sales:", df['EU_Sales'].sum())
print("Japan Sales:", df['JP_Sales'].sum())
print("Other Regions Sales:", df['Other_Sales'].sum())
print("Global Sales:", df['Global_Sales'].sum())

# Plot genre-wise sales in different regions
def plot_genre_sales(region, title, color):
    genre_sales = df.groupby('Genre')[region].sum().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    genre_sales.plot(kind='bar', color=color)
    plt.title(title)
    plt.ylabel("Sales in Million Dollars")
    plt.xlabel("Genre")
    plt.show()

plot_genre_sales('NA_Sales', "Genre-wise Sales in North America", 'skyblue')
plot_genre_sales('EU_Sales', "Genre-wise Sales in Europe", 'orange')
plot_genre_sales('JP_Sales', "Genre-wise Sales in Japan", 'skyblue')

# Year-wise sales in different regions (2001 - 2016)
def plot_yearwise_sales(region, title, color):
    yearwise_sales = df.groupby('Year')[region].sum().loc[2001:2016]
    plt.figure(figsize=(12, 6))
    yearwise_sales.plot(kind='bar', color=color)
    for i, v in enumerate(yearwise_sales):
        plt.text(i, v + 0.5, f'{v:.2f}', ha='center')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel('Total Sales (in millions)')
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

plot_yearwise_sales('NA_Sales', 'Yearly Sales in North America (2001 - 2016)', 'green')
plot_yearwise_sales('EU_Sales', 'Yearly Sales in Europe (2001 - 2016)', 'orange')
plot_yearwise_sales('JP_Sales', 'Yearly Sales in Japan (2001 - 2016)', 'skyblue')
plot_yearwise_sales('Global_Sales', 'Yearly Sales Globally (2001 - 2016)', 'turquoise')

# Top 10 games by sales in different regions
def plot_top_10_sales(region, title, color):
    top_10_sales = df.groupby('Name')[region].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    top_10_sales.plot(kind='barh', color=color)
    plt.title(title)
    plt.xlabel('Total Sales (in millions)')
    plt.ylabel('Games')
    plt.show()

plot_top_10_sales('NA_Sales', 'Top 10 Games Sales in North America', 'green')
plot_top_10_sales('EU_Sales', 'Top 10 Games Sales in Europe', 'orange')
plot_top_10_sales('JP_Sales', 'Top 10 Games Sales in Japan', 'skyblue')
plot_top_10_sales('Global_Sales', 'Top 10 Games Sales Globally', 'turquoise')
