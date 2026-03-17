import streamlit as st
import pandas as pd

st.set_page_config(layout='wide', page_title='Retail Sales Dashboard')
st.title('Retail Sales Prediction & Dashboard')

st.markdown('Upload `RetailSales.csv` if not already present in the working directory.')


df = None

uploaded = st.file_uploader('Upload CSV', type=['csv'])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    try:
        df = pd.read_csv('RetailSales.csv')   
    except Exception:
        st.error('No dataset found. Please upload RetailSales.csv.')
        st.stop()


df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Total_Sales'] = pd.to_numeric(df['Total_Sales'], errors='coerce')


st.sidebar.header('Controls')
view = st.sidebar.selectbox('View', ['Sales Trend','Category Insights','Forecast'])


if view == 'Sales Trend':
    st.subheader('Daily Sales Trend')
    daily = df.groupby('Date')['Total_Sales'].sum().asfreq('D').fillna(0)
    st.line_chart(daily)

    st.subheader('Monthly Sales')
    monthly = daily.resample('M').sum()
    st.bar_chart(monthly)


elif view == 'Category Insights':
    st.subheader('Category total sales')
    cat = df.groupby('Category')['Total_Sales'].sum().sort_values(ascending=False)
    st.bar_chart(cat)

    st.subheader('Top products')
    top = df.groupby('Product_Name')['Total_Sales'].sum().sort_values(ascending=False).head(10)
    st.table(top)


else:
    st.subheader('Forecast (precomputed if available)')
    try:
        fc = pd.read_csv('forecast.csv', index_col=0, parse_dates=True)
        st.line_chart(fc['Forecast'])
        st.write(fc.head())
    except Exception:
        st.info('No forecast file found. Run the notebook to generate forecast.csv')

