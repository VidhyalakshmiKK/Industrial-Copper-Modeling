# ========================================   /  Required Libraries   /   =================================== #
import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
import pickle

# ========================================   /   Dash board   /   =================================== #

st.set_page_config(page_title="Industrial Copper Modeling", page_icon="https://m.foolcdn.com/media/dubs/images/Industry_business_production_and_heavy_metallu.width-880.jpg",
                   layout="wide", initial_sidebar_state="auto", menu_items=None)

st.title("Industrial Copper Modeling")

tab1,tab2,tab3 =st.tabs(['SELLING PRICE PREDICTION','STATUS PREDICTION','ABOUT THE PROJECT'])

# Sidebar

with st.sidebar: 
   st.image("D:\data science - guvi\MDT-34\capstone project\Project 4 - Industrial Copper Modelling\images for streamlit\copper.png",use_column_width=True)
   st.markdown("#### Domain : :grey[Manufacturing]")
   st.markdown("#### Skills take away from this Project : :grey[Python scripting, Data Preprocessing , EDA, Streamlit]")
   st.markdown("#### Overall view : :grey[Building an application with Streamlit, using various machine learning algorithms to build a model to predict the selling price or copper and Status of leads.]")
   st.markdown("#### Developed by : :grey[VIDHYALAKSHMI K K]")

# ========================================   /   Selling Price Prediction   /   =================================== #
country_values = [25.0, 26.0, 27.0, 28.0, 30.0, 32.0, 38.0, 39.0, 40.0, 77.0, 
                    78.0, 79.0, 80.0, 84.0, 89.0, 107.0, 113.0]

status_values = ['Won', 'Lost', 'Draft', 'To be approved', 'Not lost for AM',
                    'Wonderful', 'Revised', 'Offered', 'Offerable']
status_dict = {'Lost': 0, 'Won': 1, 'Draft': 2, 'To be approved': 3, 'Not lost for AM': 4,
                'Wonderful': 5, 'Revised': 6, 'Offered': 7, 'Offerable': 8}

item_type_values = ['W', 'WI', 'S', 'PL', 'IPL', 'SLAWR', 'Others']
item_type_dict = {'W': 0, 'WI': 1, 'S': 2, 'Others': 3, 'PL': 4, 'IPL': 5, 'SLAWR': 6}

application_values = [2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 19.0, 20.0, 22.0, 25.0, 26.0, 
                        27.0, 28.0, 29.0, 38.0, 39.0, 40.0, 41.0, 42.0, 56.0, 58.0, 
                        59.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 79.0, 99.0]

product_ref_values = [611728, 611733, 611993, 628112, 628117, 628377, 640400, 
                        640405, 640665, 164141591, 164336407, 164337175, 929423819, 
                        1282007633, 1332077137, 1665572032, 1665572374, 1665584320, 
                        1665584642, 1665584662, 1668701376, 1668701698, 1668701718, 
                        1668701725, 1670798778, 1671863738, 1671876026, 1690738206, 
                        1690738219, 1693867550, 1693867563, 1721130331, 1722207579]

  
def regression():

  # get input from users
  with st.form('Regression'): 
    col1,col2,col3 = st.columns([0.5,0.1,0.5])
    with col1:
                sp_item_date=st.date_input("Item Date",key='selling_price',min_value=date(2020,7,1), 
                                        max_value=date(2021,5,31), value=date(2020,7,1))
                sp_quantity_tons=st.number_input("Quantity in tons (Min: 0.00001 & Max: 1000000000)",value=0.1)
                sp_customer_id=st.number_input("Customer ID (Min: 12458.0 & Max: 30408185.0)",value=12458.0)
                sp_country_code=st.selectbox("Country code",options=country_values)
                sp_status=st.selectbox("Status",options=status_values)
                sp_item_type=st.selectbox("Item Type",options=item_type_values)

    with col3:
                sp_application=st.selectbox("Application ",options=application_values)
                sp_thickness=st.number_input("Thickness (Min: 0.18 & Max: 2500.0)",min_value=0.1, max_value=2500.0, value=1.0)
                sp_width=st.number_input("Width (Min: 1.0 & Max: 2990.0)",min_value=1.0, max_value=2990.0, value=1.0)
                sp_product_id= st.selectbox("Product ID ",options=product_ref_values)
                sp_delivery_date=st.date_input("Delivery Date")

                st.write('')
                st.write('')
                button = st.form_submit_button(label='SUBMIT')
            
    # user entered all the input values and clicked the button
    if button:
      # load the regression pickle model
      with open(r'regression_model.pkl', 'rb') as f:
         model = pickle.load(f)
            
         # make array for all user input values in required order for model prediction
         user_data = np.array([[sp_customer_id, 
                                sp_country_code, 
                                status_dict[sp_status], 
                                item_type_dict[sp_item_type], 
                                sp_application, 
                                sp_width, 
                                sp_product_id, 
                                np.log(float(sp_quantity_tons)), 
                                np.log(float(sp_thickness)),
                                sp_item_date.day, sp_item_date.month, sp_item_date.year,
                                sp_delivery_date.day, sp_delivery_date.month, sp_delivery_date.year]])
            
         # model predict the selling price based on user input
         y_pred = model.predict(user_data)
         # inverse transformation for log transformation data
         selling_price = np.exp(y_pred[0])
         # round the value with 2 decimal point (Eg: 1.35678 to 1.36)
         selling_price = round(selling_price, 2)
         
         return selling_price

# ===========================================   / Status prediction /   ==========================================  # 

def classification():
  # get input from users
  with st.form('Classification'):
    col1,col2,col3 = st.columns([0.5,0.1,0.5])
    with col1:
      s_item_date=st.date_input("Item Date",key='status',min_value=date(2020,7,1), max_value=date(2021,5,31), value=date(2020,7,1))
      s_quantity_tons=st.number_input("Quantity in tons (Min: 0.00001 & Max: 1000000000)",value=0.1)
      s_customer_id=st.number_input("Customer ID (Min: 12458.0 & Max: 30408185.0)",value=12458.0)
      s_country_code=st.selectbox("Country code",options=country_values)
      s_selling_price=st.number_input("Selling Price (Min: 0.1 & Max: 100001015.0)",value=0.1)
      s_item_type=st.selectbox("Item Type",options=item_type_values)
    with col3:
      s_application=st.selectbox("Application ",options=application_values)
      s_thickness=st.number_input("Thickness (Min: 0.18 & Max: 2500.0)",min_value=0.1, max_value=2500.0, value=0.1)
      s_width=st.number_input("Width (Min: 1.0 & Max: 2990.0)",min_value=1.0, max_value=2990.0, value=1.0)
      s_product_id=st.selectbox("Product ID ",options=product_ref_values)
      s_delivery_date=st.date_input("Delivery Date")
      st.write('')
      st.write('')
      button = st.form_submit_button(label='SUBMIT')
      
      # user entered the all input values and click the button
      if button:
          # load the classification pickle model
          with open(r'classification_model.pkl', 'rb') as f:
              model = pickle.load(f)
          
          # make array for all user input values in required order for model prediction
          user_data = np.array([[s_customer_id, 
                              s_country_code, 
                              item_type_dict[s_item_type], 
                              s_application, 
                              s_width, 
                              s_product_id, 
                              np.log(float(s_quantity_tons)), 
                              np.log(float(s_thickness)),
                              np.log(float(s_selling_price)),
                              s_item_date.day, s_item_date.month, s_item_date.year,
                              s_delivery_date.day, s_delivery_date.month, s_delivery_date.year]])
          
          # model predict the status based on user input
          y_pred = model.predict(user_data)
          # we get the single output in list, so we access the output using index method
          status = y_pred[0]
          return status

with tab1:

    try:
    
        selling_price = regression()

        if selling_price:
            # apply custom css style for prediction text
            #style_prediction()
            st.markdown(f'### <html><body><h1 style="font-family:Sans-serif; font-size:40px"> Predicted Price : $ {selling_price} </h1></body></html>', unsafe_allow_html=True)
            st.balloons()
    

    except ValueError:

        col1,col2,col3 = st.columns([0.26,0.55,0.26])

        with col2:
            st.warning('##### Quantity Tons / Customer ID is empty', icon="⚠️")

with tab2:

    try:

        status = classification()

        if status == 1:
            
            # apply custom css style for prediction text
            hide_streamlit_style = """ <html><body><h1 style="font-family:Sans-serif; font-size:40px"> Predicted Status : Won </h1></body></html>"""
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)
            st.balloons()
            

        elif status == 0:
            
            # apply custom css style for prediction text
            #style_prediction()'
            hide_streamlit_style = """ <html><body><h1 style="font-family:Sans-serif; font-size:40px"> Predicted Status : Loss </h1></body></html>"""
            st.markdown(hide_streamlit_style, unsafe_allow_html=True)
            
    except ValueError:

        col1,col2,col3 = st.columns([0.15,0.70,0.15])

        with col2:
            st.warning('##### Quantity Tons / Customer ID / Selling Price is empty', icon="⚠️")

# ===========================================   / About Page /   ==========================================  # 

with tab3:
  col1,col2 = st.columns([3,2],gap="medium")
  with col1:
      st.write(" ")
      st.write(" ")
      st.markdown("#### **About the application :**")
      st.write( "###### <li> :grey[Industrial Copper Modeling application helps to predict the selling price and the likelihood of converting sales leads in the copper industry. By analyzing historical data on sales, market trends, and customer behavior, the application aims to assist the sales teams in pricing strategies and lead management.]",unsafe_allow_html=True)
      st.markdown("#### **Why this application ??**")
      st.write("###### <li> Market Volatility: :grey[ Copper prices fluctuate due to global demand and economic conditions, making accurate pricing crucial for profitability.]<br><br><li> Complex Pricing: :grey[Effective pricing requires analyzing multiple factors like market trends and customer relationships; a data-driven approach enhances consistency.]<br><br><li>Lead Management: :grey[Prioritizing leads is challenging; predicting conversion likelihood helps sales teams focus on high-potential opportunities.]<br><br><li> Profit Maximization: :grey[Accurate pricing and lead management prevent lost revenue and improve margins.]<br><br><li>Resource Optimization: :grey[Automating pricing and lead prediction reduces manual effort, allowing sales teams to focus on strategic tasks.] ",unsafe_allow_html=True)  
      
  with col2:
      st.write(" ")
      st.write(" ")
      st.write(" ")
      st.write(" ")
      st.write(" ")
      st.write(" ")
      st.image(r"D:\data science - guvi\MDT-34\capstone project\Project 4 - Industrial Copper Modelling\images for streamlit\3.png")
            

# ========================================   /  Completed /   =================================== #

    


#
#