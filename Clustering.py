import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from adjustText import adjust_text
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('updated_imputed_file.csv')

# Exclude Tesla as a single entity and separate its models
tesla_models = ['Model 3', 'Model Y', 'Model S', 'Model X']

# Filter and prepare the data with year information
data = df.copy()
data['Make_Model'] = data['Make'] + ' ' + data['Model Year'].astype(str)
data.loc[df['Model'].isin(tesla_models), 'Make_Model'] = data['Model'] + ' ' + data['Model Year'].astype(str)

# Calculate mean values per Make_Model
model_mean = data.groupby('Make_Model')[['Electric Range', 'Sale Price']].mean().reset_index()
model_sales = data['Make_Model'].value_counts().reset_index()
model_sales.columns = ['Make_Model', 'Sales']

# Merge the data
final_data = pd.merge(model_mean, model_sales, on='Make_Model')

# Keep top 20 Make_Model by sales
top_20 = final_data.nlargest(20, 'Sales')

# Prepare data for clustering
X = final_data[['Electric Range', 'Sale Price']]

# Standardize the features before clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
final_data['Cluster'] = kmeans.fit_predict(X_scaled)

# Add cluster labels back to the dataset
final_data['Cluster_Label'] = final_data['Cluster']

# Adjust size of the circles by scaling down
scaled_sales = final_data['Sales'] / max(final_data['Sales']) * 500  # Scale to a reasonable size

# Plot the scatter plot with clusters
plt.figure(figsize=(14, 10))
for cluster in final_data['Cluster'].unique():
    cluster_data = final_data[final_data['Cluster'] == cluster]
    plt.scatter(
        cluster_data['Electric Range'],
        cluster_data['Sale Price'],
        s=cluster_data['Sales'] / max(final_data['Sales']) * 500,  # Adjusted size
        alpha=0.7,
        edgecolor='black',
        label=f'Cluster {cluster}'
    )

# Add labels for top 20 Make_Model by sales
texts = []
for i, row in top_20.iterrows():
    texts.append(plt.text(row['Electric Range'], row['Sale Price'], row['Make_Model'], fontsize=7, alpha=0.8))

# Adjust overlapping text
adjust_text(texts, arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

# Add title and labels
plt.title('Average Sale Price vs Electric Range by Model and Year with Clusters', fontsize=16)
plt.xlabel('Average Electric Range (miles)', fontsize=14)
plt.ylabel('Average Sale Price ($)', fontsize=14)
plt.grid(True)
plt.legend()

# Show the plot
plt.show()
from sklearn.metrics import silhouette_score

# Calculate SSE for different values of k
sse = []
for k in range(1, 10):  # Test k from 1 to 9
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot SSE to find the "elbow"
plt.figure(figsize=(8, 5))
plt.plot(range(1, 10), sse, marker='o')
plt.title('Elbow Method: Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.grid(True)
plt.show()