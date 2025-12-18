import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime

st.set_page_config(
    page_title="car price prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

######### df preprocessing ############

class PrepData():
    def __init__(self, df, feature_cols=None):
        self.df = df
        self.feature_cols = feature_cols
        pass
  # maybe change csv_file path -> df in app

    def get_numeric(self, x):
        s = str(x).lower()
        pat = r"\d+\.?\d*"
        m = re.search(pat, s)
        if not m:
            return np.nan
        else:
            return float(m.group(0))

    def parse_torque_row(self, s: str):
        s = str(s).lower()

        # убираем разделители тысяч: 1,900 -> 1900
        s = re.sub(r'(\d),(?=\d{3}\b)', r'\1', s)

        # 1) момент: первое число (с точкой или без)
        m_torque = re.search(r'\d+(?:\.\d+)?', s)
        torque = float(m_torque.group()) if m_torque else np.nan

        # 2) единицы момента: nm или kgm (если нет — None)
        m_unit = re.search(r'(nm|kgm)', s)
        unit = m_unit.group(1) if m_unit else np.nan

        # 3) обороты: число или диапазон ПЕРЕД "rpm"
        m_rpm = re.search(
            r'(\d+)(?:\s*-\s*(\d+))?(?=[^0-9]*rpm\b)',
            s
        )
        if m_rpm:
            rpm_from = int(m_rpm.group(1))
            rpm_to = int(m_rpm.group(2)) if m_rpm.group(2) else rpm_from
        else:
            rpm_from = rpm_to = np.nan
        if unit == "kgm":
                torque = round(torque*9.80665,2)
        return torque, rpm_to

    def preproc_df(self):
        df = self.df.copy()
        df = df.loc[:, ~df.columns.str.match(r"(?i)^unnamed")]

        # drop duplicates
        cols = df.copy().drop(['selling_price'], axis=1, errors='ignore').columns
        df = df.drop_duplicates(subset=cols, keep='first')

        # process mileage, engine, max_power columns, max_torque_rpm, torque_num (get numbers from srt)
        for col in ['mileage', 'engine', 'max_power']:
            if col in df.columns:
                df[f'{col}_num'] = df[col].apply(self.get_numeric)
                df = df.drop(col, axis=1, errors='ignore')

        if 'torque' in df.columns:
            tmp = df['torque'].apply(self.parse_torque_row)
            df['torque_num'] = tmp.apply(lambda t: t[0])
            df['max_torque_rpm'] = tmp.apply(lambda t: t[1])
        df = df.drop('torque', axis=1, errors='ignore')
        
        if 'seats' in df.columns:
            df['seats'] = df['seats'].astype(object)
        
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        self.df = df
        return self

    
    def feature_engineering(self):
        df = self.df.copy()
        current_year = datetime.now().year

        df['power_per_liter'] = df['max_power_num'] / (df['engine_num']+1e-7)
        df['engine_power_mul'] = df['engine_num'] * df['max_power_num']
        df['km_per_year'] = df['km_driven'] / (current_year - df['year'] + 1)
        df['year_sq'] = df['year'] ** 2
        
        if 'selling_price' in df.columns:
            df['selling_price'] = np.log1p(df['selling_price'])
        
        self.df = df
        return self
    
    def preproc_df_cat(self):
        df = self.df.copy()

        df['car_label'] = df['name'].apply(lambda x: str(x).lower().split(' ')[0])
        df = df.drop('name', axis=1)
        
        cols = [
            'car_label',
            'fuel', 'seller_type', 'transmission', 'owner', 'seats']
        
        df_ohe = pd.get_dummies(df, columns=cols, drop_first=True)
        
        if self.feature_cols is None:
            self.feature_cols = df_ohe.columns  # сохраняем по трейну
        else:
            df_ohe = df_ohe.reindex(columns=self.feature_cols, fill_value=0)

        self.df = df_ohe
        return self


    def process(self):
        return self.preproc_df().feature_engineering().preproc_df_cat().df

    def x_y_data(self):
        df = self.df.copy()
        if 'selling_price' in df.columns:
            x = df.drop('selling_price', axis=1)
            y = df['selling_price']

        else:
            x = df.copy()
            y = None
            
        return x, y 

######### end of df preprocessing #####

@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_resource
def load_bundle(path="models/model.pkl"):
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "model.pkl")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    return bundle["model"], bundle["feature_cols"]
#### ui

st.title("car price prediction")

model, feature_cols = load_bundle()

uploaded_file = st.file_uploader("load data", type=["csv"])

if uploaded_file:
    df_raw = load_data(uploaded_file)

    prep = PrepData(df_raw, feature_cols=feature_cols)
    df_ready = prep.process()

    X, _ = prep.x_y_data()
    X = X.reindex(columns=feature_cols, fill_value=0)
    X = X.drop(columns=['selling_price'], errors='ignore')

    st.subheader("processed data")
    st.dataframe(X.head(20), use_container_width=True)

    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)

    df_pred = pd.DataFrame(y_pred, columns=['price prediction'])

    st.subheader("predictions")
    st.dataframe(df_pred, use_container_width=True)

    st.download_button(
        "download predictions",
        data=df_pred.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
else:
    st.info("Download csv to get predictions")