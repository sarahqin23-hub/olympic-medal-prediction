# olympic-medal-prediction
Predicting Olympic medal counts using GDP and historical performance.
# Olympic Medal Prediction

## Project Overview

This project investigates whether a country's Olympic medal count can be predicted using its previous medal performance and GDP. Two regression models (Linear Regression and Random Forest) are compared to evaluate their predictive performance under limited data conditions.

## Data Sources

1. Olympic medal dataset (1896–2022), downloaded from Kaggle.
2. World Bank GDP data (Indicator: NY.GDP.MKTP.CD).

Only Summer Olympic data were used. Observations prior to 1960 were excluded due to missing GDP values.

## Project Structure

data/
    raw/        # Original datasets
    processed/  # Cleaned dataset used for modeling

src/
    01_build_table.py      # Data cleaning and feature construction
    02_train_models.py     # Model training and evaluation
