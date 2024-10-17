import streamlit as st
import pandas as pd
from pathlib import Path
from io import StringIO
import pickle


def convert_volume(vol_str: str) -> float | None:
    if vol_str.endswith('K'):
        return float(vol_str[:-1]) * 1000

    if vol_str.endswith('M'):
        return float(vol_str[:-1]) * 1000000

    try:
        return float(vol_str)
    except Exception as e:
        print(f'ERROR: {e}')
        return None


def preprocess_data(df):
    df_new = df.sort_values('Date').copy()

    df_new['Date'] = pd.to_datetime(df_new['Date'])

    columns_to_convert = ['Price', 'Open', 'High', 'Low']

    for column in columns_to_convert:
        df_new[column] = pd.to_numeric(df_new[column].str.replace(',', ''), errors='coerce')

    df_new['Vol.'] = df_new['Vol.'].apply(convert_volume)   

    df_new['Change %'] = df_new['Change %'].str.rstrip('%').astype(float)

    idx = pd.date_range(start=df_new['Date'].min(), end=df_new['Date'].max())
    df_reindexed = df_new.set_index('Date').reindex(idx)
    df_new = df_reindexed.interpolate(method='linear')
    df_new = df_new.reset_index().rename(columns={'index': 'Date'})

    return df_new


def categorize_change(value: float | int | None) -> str:
    return 1 if value > 0 else 0


def feature_engineering(df):
    df_new = df.copy()
    df_new['Price_Diff'] = df_new['Price'] - df_new['Open']

    df_new['Price_Range'] = df_new['High'] - df_new['Low']

    df_new['Daily_Return'] = (df_new['Price'] - df_new['Open']) / df_new['Open'] * 100

    df_new['Cumulative_Return'] = (1 + df_new['Daily_Return'] / 100).cumprod() - 1

    df_new['Volatility_5d'] = df_new['Price'].rolling(window=5).std()

    df_new['Mean_5d'] = df_new['Price'].rolling(window=5).mean()

    df_new['Volume_Change'] = df_new['Vol.'].pct_change() * 100

    df_new['Volume_Mean_5d'] = df_new['Vol.'].rolling(window=5).mean()

    df_new['RoC_5d'] = df_new['Price'].pct_change(periods=5) * 100

    df_new['Bollinger_Mid'] = df_new['Price'].rolling(window=20).mean()
    df_new['Bollinger_Upper'] = df_new['Bollinger_Mid'] + (df_new['Price'].rolling(window=20).std() * 2)
    df_new['Bollinger_Lower'] = df_new['Bollinger_Mid'] - (df_new['Price'].rolling(window=20).std() * 2)

    df_new['Lagged_Price_1d'] = df_new['Price'].shift(1)
    df_new['Lagged_Volume_1d'] = df_new['Vol.'].shift(1)

    df_new['Day_of_Week'] = df_new['Date'].dt.dayofweek
    df_new['Month'] = df_new['Date'].dt.month
    df_new['Quarter'] = df_new['Date'].dt.quarter

    df_new['Target'] = df_new['Change %'].shift(-1)

    for i in range(5, 31, 5):
        df_new[f'Volatility_{i}d'] = df_new['Price'].rolling(window=i).std()
        df_new[f'Mean_{i}d'] = df_new['Price'].rolling(window=i).mean()
        df_new[f'Volume_Mean_{i}d'] = df_new['Vol.'].rolling(window=i).mean()
        df_new[f'RoC_{i}d'] = df_new['Price'].pct_change(periods=i) * 100

    for i in range(1, 31):
        df_new[f'Lagged_Change_{i}d'] = df_new['Change %'].shift(i).apply(categorize_change)

    for i in range(5, 31, 5):
        df_new[f'Change_Median_{i}d'] = df_new['Change %'].rolling(window=i).median().round(2)
        df_new[f'Change_Min_{i}d'] = df_new['Change %'].rolling(window=i).min().round(2)
        df_new[f'Change_Max_{i}d'] = df_new['Change %'].rolling(window=i).max().round(2)
        df_new[f'Change_Range_{i}d'] = (df_new[f'Change_Max_{i}d'] - df_new[f'Change_Min_{i}d']).round(2)
    
    return df_new


def categorize_change1(value: float | int | None) -> str:
    return 'Up' if value > 0 else 'Down'


def clean_data(df):
    df_new = df.copy()
    df_new = df_new.set_index('Date').round(2)
    df_new = df_new.drop(['Price', 'Open',  'High', 'Low', 'Vol.', 'Change %'], axis=1)

    df_new = df_new.dropna()
    df_new['Target'] = df_new['Target'].apply(categorize_change1)

    # cols = df_new.columns.tolist()
    # cols.remove('Target')
    # cols.append('Target')
    # df_new = df_new[cols]

    df_new = df_new.drop(['Target'], axis=1)

    return df_new




# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title = 'TrendCaster',
    page_icon = ':chart_with_upwards_trend:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------
# Declare some useful functions.

@st.cache_data
def get_data():
    df_raw = pd.read_csv(Path(__file__).parent/'data/GLO Historical Data.csv')
    df_raw['Date'] = pd.to_datetime(df_raw['Date'])
    df_raw = df_raw.sort_values('Date')
    return df_raw

df_raw = get_data()

df_raw_monthly = df_raw.groupby(pd.Grouper(key='Date', freq='M')).last()
df_raw_monthly.index = df_raw_monthly.index.strftime('%Y-%m')
df_raw_monthly = df_raw_monthly.reset_index()
df_raw_monthly = df_raw_monthly[::-1]

df_raw['Year'] = df_raw['Date'].dt.year
df_raw_yearly = df_raw.groupby('Year').last()
df_raw_yearly = df_raw_yearly.drop('Date', axis=1)
df_raw_yearly = df_raw_yearly.reset_index()

# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
'''
# :chart_with_upwards_trend: TrendCaster

## Forecast every trend with precision
'''
st.header('Globe Telecom Inc (GLO)', divider='gray')

min_value = df_raw_yearly['Year'].iloc[0]
max_value = df_raw_yearly['Year'].iloc[-1]

from_year, to_year = st.slider(
    'Yearly Trend',
    min_value=min_value,
    max_value=max_value,
    value=[min_value, max_value])

filtered_df_raw_yearly = df_raw_yearly[
    (df_raw_yearly['Year'] >= from_year) 
    & 
    (df_raw_yearly['Year'] <= to_year)
]

st.line_chart(
    filtered_df_raw_yearly[::-1],
    x='Price',
    y='Year',
)

'Glo Historical Data'
st.dataframe(df_raw[::-1])

''
''
'REMEMBER: The data you uploaded or are about to upload must contain 31 days of data for the forecast to work correctly.'
'AND MUST BE FROM Globe Telecom Inc (GLO) data only.'

uploaded_files = st.file_uploader(
    "Choose a CSV file", 
    accept_multiple_files=False,
    type={"csv", "txt"},
)

if uploaded_files is not None:
    my_df = pd.read_csv(uploaded_files)
    df_new = preprocess_data(my_df)
    df_new = feature_engineering(df_new)
    df_new = clean_data(df_new)
    df_final = df_new.iloc[[-1]]
    st.write("Your Data:")
    st.dataframe(my_df)

    loaded_model = pickle.load(open('rfc_model.sav', 'rb'))

    forecast = loaded_model.predict(df_final)
    st.write(f'Forecast: {forecast}')
    





