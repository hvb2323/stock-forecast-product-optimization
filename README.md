# Stock Forecast and Product Optimization Project

This project leverages deep learning techniques to forecast stock levels and optimize product performance based on key economic and industry growth indicators. The model predicts sales potential and provides actionable insights for product optimization.

## Project Overview

The project aims to predict the **Sales Potential** of different variants of a product using historical data. The model combines economic and industry growth rates along with seasonal factors to generate accurate forecasts. 

Key features of the project:
- **Data Preprocessing**: One-hot encoding of categorical variables (Month, Variant, Seasonality Factor) and normalization of numerical features (Industry Growth Rate and Economic Index).
- **Model**: A deep learning model built using TensorFlow and Keras for regression tasks.
- **Output**: The model predicts sales potential, helping optimize stock levels and product variants.

## Requirements

To run this project, you need the following:
- Python 3.6 or higher
- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn` (optional for visualizations)
