import streamlit as st
import pandas as pd
import joblib

lr_model = joblib.load("notebook/model.joblib")

def predict_species(sep_len, sep_wid, pet_len, pet_wid):
    d = [
        {
            "sepal_length": sep_len,
            "sepal_width": sep_wid,
            "petal_length": pet_len,
            "petal_width": pet_wid
        }
    ]
    xnew = pd.DataFrame(d)
    preds = lr_model.predict(xnew)
    probs = lr_model.predict_proba(xnew)
    classes = lr_model.classes_
    probs_dct = {}
    for c, p in zip(classes, probs.flatten()):
        probs_dct[c] = float(p)
    return preds, probs_dct

if __name__ == "__main__":
    # Initialized web page
    st.set_page_config(page_title="Iris Project", page_icon="🛠")
    st.title("Iris End to End project")
    st.subheader("By Utkarsh Gaikwad")

    # Take sep len, sep wid as input
    sep_len = st.number_input("Sepal Length", min_value=0.00, step=0.01)
    sep_wid = st.number_input("Sepal Width", min_value=0.00, step=0.01)
    pet_len = st.number_input("Petal Length", min_value=0.00, step=0.01)
    pet_wid = st.number_input("Petal Width", min_value=0.00, step=0.01)

    # Add a button to predict
    button = st.button("predict")

    # If button is clicked
    if button:
        pred, prob = predict_species(sep_len, sep_wid, pet_len, pet_wid)
        st.subheader(f"Species Predicted : {pred[0]}")
        for s, p in prob.items():
            st.subheader(f"{s} : Probability {p:.4f}")
            st.progress(p)
