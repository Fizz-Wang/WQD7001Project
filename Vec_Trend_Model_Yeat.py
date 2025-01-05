import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('updated_imputed_file.csv')

# 1. Group data by Model Year and Make to calculate the vehicle count
vehicle_counts = df.groupby(['Model Year', 'Make']).size().reset_index(name='Vehicle Count')

# 2. Get the list of all brands (Make)
brands = vehicle_counts['Make'].unique()
# 3. Set up the plot
plt.figure(figsize=(12, 8))

# Loop through each brand and plot its trend
for brand in brands:
    brand_data = vehicle_counts[vehicle_counts['Make'] == brand]
    plt.plot(
        brand_data['Model Year'],  # X-axis: Model Year
        brand_data['Vehicle Count'],  # Y-axis: Vehicle Count
        marker='o',  # Add markers to each point
        label=brand  # Label for the legend
    )

# 4. Customize the x-axis to show whole years only
plt.xticks(
    ticks=vehicle_counts['Model Year'].unique(),  # Ensure ticks are only at whole years
    labels=vehicle_counts['Model Year'].unique().astype(int)  # Convert to integer for cleaner labels
)

# 5. Add title, labels, and grid
plt.title('Vehicle Count Trend by Model Year for Each Brand', fontsize=16)
plt.xlabel('Model Year', fontsize=12)
plt.ylabel('Vehicle Count', fontsize=12)
plt.legend(title='Brands', bbox_to_anchor=(1.05, 1), loc='upper left')  # Adjust legend position
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()