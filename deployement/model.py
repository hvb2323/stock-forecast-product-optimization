import pandas as pd

df=pd.read_csv("transformed_data.csv")

df.head()


#Data Preparation

df.isnull().sum()#checking for missing values


# One-hot encode categorical variables (if they exist)
df = pd.get_dummies(df, columns=['Month', 'Variant', 'Seasonality Factor'], drop_first=True)


df.columns


from sklearn.preprocessing import StandardScaler

# Normalize numerical features
scaler = StandardScaler()
df[['Industry Growth Rate (%)', 'Economic Index']] = scaler.fit_transform(df[['Industry Growth Rate (%)', 'Economic Index']])

df


# Define a target variable (example calculation)
df['Sales Potential'] = df['Industry Growth Rate (%)'] * df['Economic Index']  # Adjust as needed




df.to_csv('new_transform.csv',index=False)


df['Sales Potential']

# Define features and target variable
X = df.drop(['Sales Potential'], axis=1)  # Replace 'Sales Potential' with your actual target column
y = df['Sales Potential']                   # Target variable



from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Build the Neural Network model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),  # Input layer
    layers.Dense(64, activation='relu'),        # Hidden layer
    layers.Dense(32, activation='relu'),        # Hidden layer
    layers.Dense(1)                              # Output layer for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Predict Sales Potential
predictions = model.predict(X_test)

# Add predictions to the test set for analysis
results = X_test.copy()
results['Predicted Sales Potential'] = predictions

# Find the variant with the highest predicted sales potential
highest_sales_variant = results.loc[results['Predicted Sales Potential'].idxmax()]

print(highest_sales_variant)


model.summary()


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')


# Extract month information for index 1743
month_row_1743 = df.loc[1743, df.columns.str.startswith('Month_')]

# Get the month name
month_name = month_row_1743[month_row_1743 == 1].index[0].replace('Month_', '')
print(f"Month for index 1743: {month_name}")


# Extract variant information for index 1743
variant_row_1743 = df.loc[1743, df.columns.str.startswith('Variant_')]

# Get the variant names
variant_name = variant_row_1743[variant_row_1743 == 1].index[0].replace('Variant_', '')
print(f"Variant for index 1743: {variant_name}")


import matplotlib.pyplot as plt

# Plotting training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Aggregate predictions by variant
variant_columns = df.columns[df.columns.str.startswith('Variant_')]
results['Variant'] = results[variant_columns].idxmax(axis=1).str.replace('Variant_', '')

# Group by variant and calculate the average predicted sales potential
forecast_df = results.groupby('Variant')['Predicted Sales Potential'].mean().reset_index()

# Plotting the forecast results
plt.figure(figsize=(12, 6))
plt.bar(forecast_df['Variant'], forecast_df['Predicted Sales Potential'], color='skyblue')
plt.title('Predicted Sales Potential by Variant')
plt.xlabel('Variant')
plt.ylabel('Predicted Sales Potential')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


