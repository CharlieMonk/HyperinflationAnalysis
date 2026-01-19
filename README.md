# Hyperinflation Analysis

Tools for analyzing stock market performance in countries experiencing hyperinflation or currency crises.

## Countries Covered

- Argentina (2018-present)
- Turkey (2018-present)
- Brazil (1995-2003)
- Russia (2014-2022)
- Mexico (1994-2002)
- Indonesia (1997-2005)

## Features

- Fetches historical data from Yahoo Finance, World Bank, and FRED
- Compares stock indices against gold, silver, and local currency depreciation
- Interactive Plotly visualizations
- Data caching for offline analysis

## Usage

Open `hyperinflation_analysis.ipynb` in Jupyter and run the cells. The static version on github does not support the interactive charts, so I recommend running it locally (or colab).

## Requirements

- pandas
- plotly
- yfinance
- pandas-datareader
