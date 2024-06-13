import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
# Load the dataset
data=pd.read_csv('OnlineRetail.csv', encoding="latin1")
print(data.head())
# print(data.shape)
# print(data.columns)
data_null=round(100*(data.isnull().sum())/len(data), 2)
data=data.dropna()
# print(data_null)
# print(data.describe())
# data.drop(['StockCode'], axis=1, inplace=True)
# print(data.columns)
data['CustomerID']=data['CustomerID'].astype(str)
print(data.head())
data['Amount']=data['Quantity']*data['UnitPrice']
print(data.info())
data_m = data.groupby('CustomerID')['Quantity'].sum().sort_values(ascending=False)
print(data_m.head())
# product with the highest sales
data_m = data.groupby('Description')['Amount'].sum().sort_values(ascending=False)
print(data_m.head())
# country with the highest product quantity sold
data_m = data.groupby('Country')['Quantity'].sum().sort_values(ascending=False)
print(data_m.head())
# most frequent product sold
data_m = data.groupby('Description')['Quantity'].count().sort_values(ascending=False)
print(data_m.head())
# total sales for the last month
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], format='%m/%d/%Y %H:%M')
print(data['InvoiceDate'].head())
max_date=max(data['InvoiceDate'])
print(max_date)
min_date=min(data['InvoiceDate'])
print(min_date)
# total number of days
difference=(max_date-min_date)
print(difference)
# Convert the provided date string to a datetime object
provided_date = pd.to_datetime('2011-12-09 12:50:00')

# Calculate the date one month before the provided date
one_month_before = provided_date - pd.DateOffset(months=1)

# Filter the data for transactions within one month before the provided date, including the provided date
one_month_before_data = data[(data['InvoiceDate'] >= one_month_before) & (data['InvoiceDate'] <= provided_date)]

# Calculate the total sales for one month before the provided date
total_sales_one_month_before = one_month_before_data['Amount'].sum()

print("Total Sales for One Month Before", provided_date, ":", total_sales_one_month_before)
# Therefore the total sales for the last one one is Shs 1156014.39
data = data[(data['Quantity'] > 0) & (data['UnitPrice'] > 0)]
# Sample data for faster processing
sample_data = data.sample(frac=0.1, random_state=0)
# Select features for clustering
X = sample_data[['Quantity', 'UnitPrice']]
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Define a range for K and an empty list to store silhouette scores
K = range(2, 7)
# Use the elbow method to check the optimal value of K
ssd = []
for k in K:
    kmeans = KMeans(n_clusters=k, max_iter=50)
    kmeans.fit(X_scaled)

    ssd.append(kmeans.inertia_)
print("\nSSD: ", ssd)
# plot the SSDs for each n_clusters
plt.plot(K, ssd, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Sum of Squared Distances (SSD)")
plt.title("Elbow Method for Optimal k Value")
plt.show()
print("\nThe optimal value of k is 3.")

"""silhouette_scores = []
# Calculate and print silhouette scores for each K
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    silhouette_scores.append(score)
    print(f"K={k}, Silhouette Score={score:.4f}")
# Plot the silhouette scores
sns.lineplot(x=K, y=silhouette_scores)
plt.show()
# Conclusion
'''Based on the silhouette score analysis of the online retail dataset, the optimal number of clusters is two
determined by the highest silhouette score'''"""
