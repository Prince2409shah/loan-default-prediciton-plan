import streamlit as st
import pandas as pd
import joblib
model=joblib.load('xgboost_credit_risk_pipeline_final.joblib')
st.title("Loan Default Prediction")

st.write(
    "This application predicts the probability that a loan applicant will default "
    "based on their financial and credit information."
)

st.sidebar.title("Applicant Information")
ext_score_2=st.sidebar.number_input('External Credit Score 2',min_value=0.0,max_value=1.0,value=0.5)
education_type=st.sidebar.selectbox('Education Type',['Secondary / secondary special', 'Higher education',
       'Incomplete higher', 'Lower secondary', 'Academic degree'])
income_type=st.sidebar.selectbox('Income Type',['Working', 'State servant', 'Commercial associate', 'Pensioner',
       'Unemployed', 'Student', 'Businessman', 'Maternity leave'])
ext_score_3=st.sidebar.number_input('External Credit Score 3',min_value=0.0,max_value=1.0,value=0.5)
owns_car=st.sidebar.selectbox('Own Car',['N', 'Y'])
goods_price=st.sidebar.number_input("Goods Price",min_value=0,value=100000)
gender = st.sidebar.selectbox("Gender (Optional)", ["Not Provided", "M", "F", "XNA"])
if gender == "Not Provided":
    gender = None
input_data={
    'EXT_SOURCE_2':ext_score_2,
    'NAME_EDUCATION_TYPE':education_type,
    'NAME_INCOME_TYPE':income_type,
    'EXT_SOURCE_3':ext_score_3,
    'FLAG_OWN_CAR':owns_car,
    'AMT_GOODS_PRICE':goods_price,
    'CODE_GENDER':gender,
} 
input_df=pd.DataFrame([input_data])
st.write('Input Data Preview')
st.dataframe(input_df)


if st.button("Predict Default Risk", key="predict_"):
    prob = model.predict_proba(input_df)[0][1]
    st.write(f"Default Probability: {prob:.2f}")

    if prob < 0.1:
        st.success("Low Default Risk")
    elif prob < 0.2:
        st.warning("Medium Default Risk")
    else:
        st.error("High Default Risk")

