import pandas as pd
import streamlit as st
from joblib import load
from mes_fonctions import *

labels = pd.read_csv('label_pred.csv')
def get_label(pred):
    df = pd.DataFrame(pred, columns=labels.columns.values)
    y = df.replace(0, float("NaN")).dropna(how='all', axis=1).columns.values.tolist()
    return y


st.write("""
# Application de suggestion de tags

Cette application propose une classification de question sur stackoverflow
""")


def user_input_features():
	text = st.text_area('Entrez votre question', "text")
	return preparation_data(text)

df = user_input_features()

clf = load('mon_model.joblib')

prediction = clf.predict(df)

st.subheader('Suggestion')
st.write(get_label(prediction))