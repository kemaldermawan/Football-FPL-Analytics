import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="FPL Analytics",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football - Fantasy Premier League Analytics")
st.markdown("Welcome to the FPL analytics system. The virtual environment is operating correctly.")

st.sidebar.header("Control Parameters")
st.sidebar.info("The data extraction module will be integrated in the next phase.")

st.subheader("Metric Data Testing")
dummy_data = pd.DataFrame({
    'Player': ['Erling Haaland', 'Mohamed Salah', 'Bukayo Saka'],
    'FPL Points': [14, 12, 9]
})
st.dataframe(dummy_data, use_container_width=True)