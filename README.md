# Data Lab

Data Lab is a Flask-based web application for performing data analysis, statistical operations, visualization, and regression modeling through an interactive browser interface.

## Overview

The project is designed to help users upload datasets and work with them using a single web platform. It combines data modification, time series processing, descriptive statistics, inferential statistics, data visualization, and linear regression in one application.

## Features

- Upload CSV and Excel files
- Perform data modification operations such as sorting, filtering, segmentation, and column management
- Apply time series operations such as resampling, moving averages, and percentage change
- Generate descriptive statistics and correlation analysis
- Create multiple visualizations such as bar charts, line charts, scatter plots, histograms, box plots, pie charts, and heatmaps
- Perform inferential statistics including confidence intervals and t-tests
- Train and use linear regression models for prediction
- Download generated charts

## Modules

### Module 3.1: Data Modifications
Allows users to sort, filter, segment, group, and modify dataset columns.

### Module 3.2: Time Series Modifications
Supports operations like date filtering, resampling, moving averages, and percentage change analysis.

### Module 3.3: Descriptive Statistics
Provides summary statistics, correlation analysis, heatmaps, and pairplots.

### Module 3.4: Data Visualization
Generates different chart types for visual exploration of data.

### Module 3.5: Inferential Statistics
Includes confidence intervals, one-sample t-tests, and two-sample t-tests.

### Module 3.6: Linear Regression
Allows users to train OLS regression models and generate predictions from input values.

## Tech Stack

- Python
- Flask
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- scikit-learn
- joblib
- HTML
- CSS
- JavaScript

## Project Structure

```bash
data-lab/
│
├── app.py
├── README.md
│
├── templates/
│   ├── index.html
│   ├── upload_success.html
│   ├── module_3_1.html
│   ├── module_3_2.html
│   ├── module_3_3.html
│   ├── module_3_4.html
│   ├── module_3_5.html
│   └── module_3_6.html
│
├── static/
│   ├── css/
│   ├── js/
│
├── uploads/
```

## Outputs

The application can generate:

- Modified data tables
- Time series results
- Descriptive statistics summaries
- Correlation heatmaps
- Pairplots
- Data visualizations
- Confidence interval results
- T-test outputs
- Regression summaries
- Predicted values

## Future Improvements

- Add more machine learning models
- Improve UI/UX design
- Add Pre-Processing modules

## License

This project is developed for academic purposes.