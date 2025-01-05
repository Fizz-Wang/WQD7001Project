import pandas as pd

# Load the dataset
df = pd.read_csv('Electric_Vehicle_Title_and_Registration_Activity_20241104.csv')

# Calculate the mean Sale Price for each combination of Model and Model Year,
# excluding rows where Sale Price is 0. Round the mean to the nearest integer.
mean_prices = df[df['Sale Price'] > 0].groupby(['Model', 'Model Year'])['Sale Price'].mean().round()


# Define a function to impute Sale Price where it's 0, using the calculated mean.
def impute_price(row):
    if row['Sale Price'] == 0:
        # Use the mean for the corresponding Model and Model Year, or 0 if no mean exists.
        return mean_prices.get((row['Model'], row['Model Year']), 0)
    return row['Sale Price']


df = df[df['New or Used Vehicle'] == 'New']

# Apply the imputation function to the Sale Price column
df['Sale Price'] = df.apply(impute_price, axis=1)

# Remove rows where Sale Price is still less than 10000
df = df[df['Sale Price'] >= 10000]
df = df[df['Sale Price'] <= 100000]
df = df[df['Model Year'] >= 2012]
df = df[df['Model Year'] <= 2023]

# Keep only rows where the vehicle is new

# Display the total number of rows in the dataset
print(f'Total number of rows: {len(df)}')

# Save the updated dataset to a new CSV file
df.to_csv('imputed_file.csv', index=False)