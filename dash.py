import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import numpy as np
import shap
import matplotlib.pyplot as plt

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'customer_data': data}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def customer_data(data_path):
    df=pd.read_csv(data_path, nrows=50)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    customer_list = df["SK_ID_CURR"].drop_duplicates().to_list()
    return df, customer_list

def gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted score"},
        gauge = {'axis': {'range': [0, 1]},
             'steps' : [
                 {'range': [0, 0.45], 'color': "darkorange"},
                 {'range': [0.45, 0.55], 'color': "yellow"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}}))

    st.plotly_chart(fig, theme="streamlit")

def main():
    FastAPI_URI = 'https://opc-p8-fastapi.herokuapp.com/predict'
    data_path = 'df_train_model_selected_50cust.csv'

    df_customer, customer_list = customer_data(data_path=data_path)

    st.title('Simulation for a customer loan request')
    selected_customer = st.text_input('Customer ID (format exemple : 200605):')
    customer_btn = st.button('Search for customer')

    if customer_btn:
        if int(selected_customer) in customer_list:
            filtered_customer = int(selected_customer)
            st.success("Selected customer : %s" %filtered_customer)
            df_filtered = df_customer[df_customer["SK_ID_CURR"]==filtered_customer]
            st.dataframe(df_filtered.drop(columns="TARGET"))

            X_cust = [i for i in df_filtered.iloc[:,2:].values.tolist()[0]]
            pred, shap_values, shap_base_value = request_prediction(FastAPI_URI, X_cust)

            # X_std = pipeline[0].transform([X_cust])
            #shap_values_calc = shap_explainer(X_std[0:1])
            shap_obj = shap.Explanation(np.array(shap_values), base_values=np.array(shap_base_value))
            plt.title('Local FI')
            shap_plot = shap.plots.waterfall(shap_obj[0])
            st.pyplot(shap_plot, bbox_inches='tight')

            gauge(1-pred[0])
            if pred[0] <= 0.5:
                st.write(
                    'Loan Decision: ACCEPTED')
            elif pred[0] > 0.5:
                st.write(
                    'Loan Decision: REFUSED')
            
        else :
            st.warning("Unknown customer")

if __name__ == '__main__':
    main()

# streamlit run dash.py