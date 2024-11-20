import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
df = pd.read_csv("new_transform.csv") 

# Define features and target variable
X = df.drop(['Sales Potential'], axis=1)  # Replace 'Sales Potential' with your actual target column
y = df['Sales Potential']                   # Target variable

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
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)
# Predict Sales Potential
predictions = model.predict(X_test)

# Add predictions to the test set for analysis
results = X_test.copy()
results['Predicted Sales Potential'] = predictions
# Extract month and variant information
month_columns = df.columns[df.columns.str.startswith('Month_')]
results['Month'] = results[month_columns].idxmax(axis=1).str.replace('Month_', '')

variant_columns = df.columns[df.columns.str.startswith('Variant_')]
results['Variant'] = results[variant_columns].idxmax(axis=1).str.replace('Variant_', '')

# Group by Month and Variant, calculating the average predicted sales potential
forecast_df1 = results.groupby(['Month', 'Variant'])['Predicted Sales Potential'].mean().reset_index()

# Streamlit dashboard layout
st.title('Sales Potential Forecast Dashboard')
# Variant selection
selected_variant = st.selectbox('Select a Variant', forecast_df1['Variant'].unique())
# Filter data for the selected variant
filtered_data = forecast_df1[forecast_df1['Variant'] == selected_variant]
# Plotting the forecast results
plt.figure(figsize=(14, 7))
plt.bar(filtered_data['Month'], filtered_data['Predicted Sales Potential'], color='skyblue')
plt.title(f'Predicted Sales Potential for {selected_variant} by Month')
plt.xlabel('Month')
plt.ylabel('Predicted Sales Potential')
plt.xticks(rotation=45)
plt.grid(axis='y')
st.pyplot(plt)
# Additional data insights
st.write("### Average Predicted Sales Potential:")
st.write(filtered_data)


