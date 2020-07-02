import streamlit as st
import datetime
import numpy as np
import pandas as pd
###########################
###########################
st.title('L')
st.write("Here's our first attempt at using data to create a table:")
st.write(pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
}))
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.line_chart(chart_data)
# Data
loaigd = pd.DataFrame({
  'first column': ['Lịch sử giá và GDNDTNN','Lịch sự kiện'],
  'second column': ['lichsugia','lichsukien']
})

#Slide bar
title = st.sidebar.title('Lịch sử giao dịch')
option = st.sidebar.radio(
    '1. Chọn loại giao dịch',
     loaigd['first column'])
date = st.sidebar.date_input('2. Ngày giao dịch', datetime.datetime.today()) 
