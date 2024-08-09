# Confidence Thapelo Makofane - InternCareer Task 1

#Import libraries
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

#Loading Dataset
data = pd.read_csv('youtubers_df.csv')

#Handling Missing Values
data.fillna({'Categories' : 'Unknown'}, inplace=True)

Missing = data.isnull().sum()

#1. Top Categories
top_categories = data['Categories'].value_counts().head(10)

#Engagement by Category
category_engagement = data.groupby('Categories')[['Visits', 'Likes', 'Comments']].mean()

#Plot of top categories
plt.figure(figsize=(10, 6))
sns.barplot(x=top_categories.values, y=top_categories.index)
plt.title('Top 10 Categories by Number of Streamers')
plt.xlabel('Number of Streamers')
plt.show()

#Plot engagement by category
plt.figure(figsize=(15, 6))
category_engagement.plot(kind='bar', stacked=True)
plt.title('Average Engagement by Category')
plt.xlabel('Category')
plt.ylabel('Average Engagement')
plt.tight_layout()
plt.show()

#2. Scatter plots to visualize Correlations
plt.figure(figsize=(15, 10))

#Scatter plot: Subscribers VS Visits
plt.subplot(2, 2, 1)
sns.scatterplot(x='Suscribers', y='Visits', data=data)
plt.title('Subscribers vs Visits')

#Scatter plot: Subscribers VS Likes
plt.subplot(2, 2, 2)
sns.scatterplot(x='Suscribers', y='Likes', data=data)
plt.title('Subscribers VS Likes')

#Scatter plot: Subscribers VS Comments
plt.subplot(2, 2, 3)
sns.scatterplot(x='Suscribers', y='Comments', data=data)
plt.title('Subscribers VS Comments')

#Scatter plot: Likes VS Comments
plt.subplot(2, 2, 4)
sns.scatterplot(x='Likes', y='Comments', data=data)
plt.title('Likes VS Comments')

plt.tight_layout()
plt.show()

#Trend Analysis
plt.figure(figsize=(15,10))

#Line chart for Subscribers over Rank
plt.subplot(2, 2, 1)
sns.lineplot(data=data, x='Rank', y='Suscribers')
plt.title('Subscribers over Rank')

#Line chart for Visits over Rank
plt.subplot(2, 2, 2)
sns.lineplot(x='Rank', y='Visits', data=data)
plt.title('Visits over Rank')

#Line chart for Likes over Rank
plt.subplot(2, 2, 3)
sns.lineplot(x= 'Rank', y='Likes', data=data)
plt.title('Likes over Rank')

#Line chart for Comments over Rank
plt.subplot(2, 2, 4)
sns.lineplot(x='Rank', y='Comments', data=data)
plt.title('Comments over Rank')

plt.tight_layout()
plt.show()

#Correlation Heatmap
plt.figure(figsize=(10, 6))
correlation_matrix = data[['Suscribers', 'Visits', 'Likes', 'Comments']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

#Calculaion of engagement rates
data['Likes_per_Visit'] = data['Likes'] / data['Visits']
data['Comments_per_Visit'] = data['Comments'] / data['Visits']

#Distribution of engagement rates
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.histplot(data['Likes_per_Visit'].dropna(), kde=True, bins=30)
plt.title('Distribution of Likes per Visit')

plt.subplot(1, 2, 2)
sns.histplot(data['Comments_per_Visit'].dropna(), kde=True, bins=30)
plt.title('Distribution of Comments per Visit')

plt.tight_layout()
plt.show()

#3. Distribution of Streamers by Country
country_distribution = data['Country'].value_counts()

#Plot distribution of Streamers by country
plt.figure(figsize=(12, 6))
sns.barplot(x=country_distribution.values, y=country_distribution.index)
plt.title('Distribution of Streamers by Country')
plt.xlabel('Number of Streamers')
plt.show()

#Regional preferences for specific content Categories
category_country_pivot = data.pivot_table(index='Country', columns='Categories',
                                          values='Username', aggfunc='count', fill_value=0)

#Plot regional preferences
plt.figure(figsize=(15, 10))
sns.heatmap(category_country_pivot, cmap='viridis')
plt.title('Regional Preferences for Content Categories')
plt.xlabel('Content Category')
plt.ylabel('Country')
plt.show()

#4. Calculate Average Performance Metrics
average_metrics = data[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

#Plot Average Performance Metrics
plt.figure(figsize=(10, 6))
average_metrics.plot(kind='bar')
plt.title('Average Performance Metrics')
plt.ylabel('Average Count')
plt.show()

#Identify Patterns or Anomalies in these metrics
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.histplot(data['Suscribers'], kde=True, bins=30)
plt.title('Distribution of Subscribers')

plt.subplot(2, 2, 2)
sns.histplot(data['Visits'], kde=True, bins=30)
plt.title('Distribution of Visits')

plt.subplot(2, 2, 3)
sns.histplot(data['Likes'], kde=True, bins=30)
plt.title('Distribution of Likes')

plt.subplot(2, 2, 4)
sns.histplot(data['Comments'], kde=True, bins=30)
plt.title('Distribution of Comments')

plt.tight_layout()
plt.show()

#5. Distribution of Content Categories
category_distribution = data['Categories'].value_counts()

#Plot Distribution of Content Categories
plt.figure(figsize=(12, 6))
sns.barplot(x=category_distribution.values, y=category_distribution.index)
plt.title('Distribution of Content Categories')
plt.xlabel('Number of Streamers')
plt.show()

#Performance Metrics by Category
category_performance = data.groupby('Categories')[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

#Plot Performance Metrics by Category
category_performance.plot(kind='bar', figsize=(15, 10), stacked=True)
plt.title('Average Performance Metrics by Category')
plt.xlabel('Content Category')
plt.ylabel('Average Metrics')
plt.tight_layout()
plt.show()

#6. Brands and Collaborations - incomplete data to complete request
# Add a hypothetical 'Brand_Collaborations' column with random True/False values
np.random.seed(42)  # For reproducibility
data['Brand_Collaborations'] = np.random.choice([True, False], size=len(data))

# Calculate average metrics for streamers with and without brand collaborations
collaboration_performance = data.groupby('Brand_Collaborations')[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

# Plot average metrics for streamers with and without brand collaborations
plt.figure(figsize=(10, 6))
collaboration_performance.plot(kind='bar', stacked=True)
plt.title('Performance Metrics by Brand Collaborations')
plt.xlabel('Brand Collaborations')
plt.ylabel('Average Metrics')
plt.tight_layout()
plt.show()

#7.Benchmarking
#Calculate average performance metrics
average_performance = data[['Suscribers', 'Visits', 'Likes', 'Comments']].mean()

#Identify streamers with above-average performance
top_performers = data[(data['Suscribers'] > average_performance['Suscribers']) &
                      (data['Visits'] > average_performance['Visits']) &
                      (data['Likes'] > average_performance['Likes']) &
                      (data['Comments'] > average_performance['Comments'])]

#Display Top Performers
top_performers_summary = top_performers[['Username', 'Suscribers', 'Visits', 'Likes', 'Comments']]

#Display the top 10 Performers
print('Top Performers Summary:\n', top_performers_summary.head(10))

#8. Content recommendations based on categories and performance metrics

top_performers_by_category = (
    data.groupby('Categories', group_keys=False)
    .apply(lambda x: x.nlargest(5, 'Suscribers'))
    [['Username', 'Suscribers', 'Visits', 'Likes', 'Comments']]
)

# Display top performers by category for recommendation
print(top_performers_by_category)

