"""
Streamlit Cheat Sheet

Capstone Project DQLABS X LKPP
Created By : Alex Putra Setiadi

"""

import streamlit as st
from pathlib import Path
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
from sklearn.metrics import mean_squared_error

# Initial page config
logo = "new_lkpp.png"
title = "Capstone Project DQLABS X LKPP"

st.set_page_config(
    page_title=title,
    layout="wide",
    initial_sidebar_state="expanded",
)

file = None

def main():
    cs_sidebar()
    process_file()
    return None


# Thanks to streamlitopedia for the following code snippet

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


# sidebar

def cs_sidebar():
    st.sidebar.image(logo, width=250)
    st.sidebar.markdown('''<hr>''', unsafe_allow_html=True)
    st.sidebar.markdown('CREDIT KELOMPOK 2')
    st.sidebar.code(
        '''
            >>> Alex Putra Setiadi
            >>> Jidda Hadiyana
            >>> Ina
            >>> Iswan
            >>> Yunita Nurjanah
        ''')

    return None

#process file
def process_file():
    df = pd.read_excel("data_excel.xlsx")
    st.markdown('''### Detail Data''', unsafe_allow_html=True)
    st.write(df.head())
    col1, col2 = st.columns(2)
    col1.subheader("5 Data teratas")
    col1.write(df.head())
    col2.subheader("Describe Data")
    col2.write(df.describe())
    st.markdown('''<hr>''', unsafe_allow_html=True)
    st.markdown('''### Cleaning Data''', unsafe_allow_html=True)
    kolom_yang_diinginkan = ["pagu", "hps", "kategori_pengadaan", "nilai_penawaran", "nilai_terkoreksi",
                             "nilai_negosiasi", "nilai_kontrak"]
    df = df[kolom_yang_diinginkan]
    dataclean = df.dropna()
    percentage_null = (dataclean.isnull().sum() / len(dataclean)) * 100

    # Create a DataFrame to display the percentage of null values and sort it
    null_summary = pd.DataFrame({
        'Column': percentage_null.index,
        'Percentage_Null': percentage_null.values
    })

    # Sort the DataFrame by the Percentage_Null column in descending order
    null_summary = null_summary.sort_values(by='Percentage_Null', ascending=False)
    col1, col2 = st.columns(2)
    col1.subheader("Rangkuman Missing value")
    col1.write(df.isnull().sum())
    col1.write(null_summary)
    col2.subheader("Data setelah cleansing")
    col2.write(dataclean)
    dataclean = dataclean.drop_duplicates()
    dataclean = dataclean.dropna().reset_index()

    dataclean = dataclean.rename(columns={
        'pagu': 'PAGU',
        'hps': 'HPS',
        'kategori_pengadaan': 'KATEGORI_PENGADAAN',
        'nilai_penawaran': 'NILAI_PENAWARAN',
        'nilai_terkoreksi': 'NILAI_TERKOREKSI',
        'nilai_negosiasi': 'NILAI_NEGOSIASI',
        'nilai_kontrak': 'NILAI_KONTRAK'
    })
    st.subheader("Data Rangkuman setelah cleansing dan rename label")
    st.write(dataclean)
    label_encoder = LabelEncoder()

    column_categorical = ["KATEGORI_PENGADAAN"]
    encoded_result = label_encoder.fit_transform(dataclean[column_categorical])
    dataclean.drop(column_categorical, inplace=True, axis=1)
    encoded_dataframe = pd.DataFrame({"KATEGORI_PENGADAAN": encoded_result})
    dataclean = dataclean.merge(encoded_dataframe, left_index=True, right_index=True, how='left')
    dataclean['NILAI_NEGOSIASI'] = np.select(condlist=[dataclean['NILAI_NEGOSIASI'] == 0],
                                             choicelist=[dataclean['NILAI_TERKOREKSI']],
                                             default=dataclean['NILAI_NEGOSIASI'])
    dataclean = dataclean[dataclean['PAGU'] != 1]

    sns.histplot(dataclean['PAGU'], kde=True, bins=30, color='blue')
    plt.title('Histogram and Normal Distribution')
    dataclean.boxplot(figsize=(20, 25))
    dataclean["PAGU"]
    sns.boxplot(x=dataclean["PAGU"])
    fig, axs = plt.subplots(ncols=7, nrows=2, figsize=(20, 10))
    index = 0
    axs = axs.flatten()
    for k, v in dataclean.items():
        sns.boxplot(y=k, data=dataclean, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    st.pyplot(plt.show())

    return None

# Run main()

if __name__ == '__main__':
    main()