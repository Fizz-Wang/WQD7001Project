import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('updated_imputed_file.csv')

# 1. Calculate the top 10 models by sales volume
top_models = (
    df['Model']
    .value_counts()
    .head(10)
    .index
)

# 2. Filter the dataset to include only the top 10 models
filtered_df = df[df['Model'].isin(top_models)]

# 3. Create the boxplot for Sale Price distribution by Model
plt.figure(figsize=(12, 8))
filtered_df.boxplot(
    column='Sale Price',  # Numeric column to analyze
    by='Model',  # Grouping column
    grid=False,  # Remove grid lines
    showfliers=False  # Hide outliers for cleaner visualization
)

# 4. Customize the plot
plt.title('Sale Price Distribution for Top 10 Models', fontsize=16)
plt.suptitle('')  # Remove default subtitle
plt.xlabel('Model', fontsize=12)
plt.ylabel('Sale Price (USD)', fontsize=12)
plt.xticks(rotation=45)  # Rotate model names for better readability
plt.tight_layout()

# Show the plot
plt.show()