import pandas as pd
import streamlit as st
import requests
import plotly.graph_objects as go
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

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
    df=pd.read_csv(data_path)
    df["SK_ID_CURR"] = df["SK_ID_CURR"].astype(int)
    customer_list = df["SK_ID_CURR"].drop_duplicates().to_list()
    return df, customer_list

def gauge(score):
    # Draw a gauge to illustrate the predicted score
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Customer solvency score"},
        gauge = {'axis': {'range': [0, 1]},
             'steps' : [
                 {'range': [0, 0.48], 'color': "darkorange"},
                 {'range': [0.48, 0.58], 'color': "yellow"}],
             'threshold' : {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': score}}))
    st.plotly_chart(fig, theme="streamlit")
    # Define loan attribution choice according to predicted score
    if score > 0.47:
        decision_message = 'Loan Decision: ACCEPTED'
    elif score <=  0.47:
        decision_message = 'Loan Decision: REFUSED'
    return decision_message

def multi_features_plot(data, feat_1, feat_2, filtered_customer, display_score):
    st.subheader("Selected Customer position compared to others")
    df_cust = data[data["SK_ID_CURR"]==filtered_customer]
    df = data[[feat_1, feat_2, 'y_pred', 'y_score']]
    df_ok = df[df['y_pred']==0]
    df_ko = df[df['y_pred']==1]
    df_cust = df_cust[[feat_1, feat_2, 'y_pred']]

    fig_1 = go.Figure(
        data=[
            go.Scatter(
                x=df_ok[feat_1],
                y=df_ok[feat_2],
                visible='legendonly',
                mode='markers',
                marker_color='green',
                name="Accepted Loans"
                ),
            go.Scatter(
                x=df_ko[feat_1],
                y=df_ok[feat_2],
                mode='markers',
                visible='legendonly',
                marker_color='yellow',
                name="Rejected Loans"
                ),
            go.Scatter(
                x=df_cust[feat_1],
                y=df_cust[feat_2],
                mode="markers",
                marker=dict(
                    color="red",
                    size=15,
                    ),
                name="Selected Customer"
                )
            ]
        )
    
    fig_2 = go.Figure(
        data=[
            go.Scatter(
                x=df[feat_1],
                y=df[feat_2],
                mode='markers',
                marker=dict(
                    size=16,
                    color=df['y_score'],
                    colorscale='Viridis',
                    showscale=True
                    ),
                name="Customers score",
                ),
            go.Scatter(
                x=df_cust[feat_1],
                y=df_cust[feat_2],
                mode="markers",
                marker=dict(
                    color="red",
                    size=15,
                    ),
                name="Selected Customer"
                )
            ]
        )

    fig_1.update_layout(xaxis_title=feat_1, yaxis_title=feat_2)
    fig_2.update_layout(xaxis_title=feat_1, yaxis_title=feat_2)
    fig_2.update_layout(legend_traceorder="reversed")

    if display_score:
        st.plotly_chart(fig_2, use_container_width=True)
    else:
        st.plotly_chart(fig_1, use_container_width=True)

def global_FI_plot(n_top=20):
    # Fonction ploting n_top most important coeff of the logistic regression
    plt.figure(figsize=(15,6))
    df_coeff = pd.read_csv('df_model_selected_coef.csv')
    logistic_reg_coeff = df_coeff.loc[0].values
    color_list =  sns.color_palette("dark", len(df_coeff.columns)) 
    top_x = n_top
    idx = np.argsort(np.abs(logistic_reg_coeff))[::-1] 
    lreg_ax = plt.barh(df_coeff.columns[idx[:top_x]][::-1], logistic_reg_coeff[idx[:top_x]][::-1])
    for i,bar in enumerate(lreg_ax):
        bar.set_color(color_list[idx[:top_x][::-1][i]])
    plt.box(False) 
    plt.suptitle(
        "Global importance of main parameters (Top " + str(top_x) + ")",
         fontsize=20, fontweight="normal"
         )
    st.pyplot()

def main():
    FastAPI_URI = 'https://opc-p8-fastapi.herokuapp.com/predict'
    data_path = 'df_model_selected_1pcust.csv'

    df_customer, customer_list = customer_data(data_path=data_path)
    features_list = df_customer.iloc[:,2:-2].columns.values

    st.title('Simulation for a customer loan request')
    selected_customer = st.sidebar.text_input('Customer ID (format exemple : 309518):')
    customer_btn = st.sidebar.button('Search for customer')
    feature_select = st.sidebar.multiselect(
                        "Select 2 features to see customer position",
                        features_list, default=["AMT_GOODS_PRICE", "EXT_SOURCE_2"]
                        )
    display_score = st.sidebar.checkbox("Display score (else, prediction)", value=False)

    if customer_btn:
        if int(selected_customer) in customer_list:
            filtered_customer = int(selected_customer)
            st.success("Selected customer : %s" %filtered_customer)

            df_filtered = df_customer[df_customer["SK_ID_CURR"]==filtered_customer]
            st.dataframe(df_filtered.drop(columns=["TARGET", "y_score", "y_pred"]))

            X_cust = [i for i in df_filtered.iloc[:,2:-2].values.tolist()[0]]
            pred, shap_values, shap_base_value = request_prediction(FastAPI_URI, X_cust)

            verdict = gauge(pred[0])
            st.write(verdict)

            multi_features_plot(
                df_customer, 
                feature_select[0], feature_select[1], 
                filtered_customer, 
                display_score
                )

            shap_obj = shap.Explanation(
                np.array(shap_values), 
                base_values=np.array(shap_base_value), 
                feature_names=features_list
                )
            
            plt.title('Main parameters impacting customer %s loan decision' %filtered_customer)

            shap_plot = shap.plots.waterfall(shap_obj[0])
            st.pyplot(shap_plot, bbox_inches='tight')

            global_FI_plot()
                        
        else :
            st.warning("Unknown customer")

if __name__ == '__main__':
    main()

# streamlit run dash.py