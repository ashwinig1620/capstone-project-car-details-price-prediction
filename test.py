import pandas as pd
import pickle 
import streamlit as st 
from sklearn.preprocessing import LabelEncoder



def main():
    html_temp="""
    <div style ="background-color:blue;padding:16px">
    <h2 style="color:white;text-align:center;"> Car Price Prediction Using ML </h2>
    </div>
    """
    
    df=pd.read_csv('D:/CAR DETAILS.csv')
    

    lb=LabelEncoder()
    
    
    with open('gb_regressor_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

     

    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.write(' ')
    st.write(' ')
    
   
   
    st.markdown("##### Are you planning to sell your car? then, predict the best price of your car\n##### so let's try evaluating the price")
    
    
    
    r1 = st.selectbox("Select the car name", df['name'].unique())
    
    r2=st.number_input(" what is the distance completed by the car in Kilometers?",100,500000,step=100)
    
    r3=st.selectbox("what is the fuel type of the car?",df['fuel'].unique())
       
    r4=st.selectbox("Are you Individual or Dealer or Trustmark Dealer?",df['seller_type'].unique())

    r5=st.selectbox("What is the Transmission Type of your car?",df['transmission'].unique())
    
    r6=st.selectbox("Number of Owners the car Previously had?",df['owner'].unique())
    
    r7=st.slider(" In which year car was purchased?",1990,2023)
        
    
    data_new=pd.DataFrame({
        'name':r1,
        'km_driven':r2,
        'fuel':r3,
        'seller_type':r4,
        'transmission':r5,
        'owner':r6,
        'age':r7
    
    },index=[0])
       
    data_new['name']=lb.fit_transform(data_new['name'])
    data_new['fuel']=lb.fit_transform(data_new['fuel'])
    data_new['seller_type']=lb.fit_transform(data_new['seller_type'])
    data_new['transmission']=lb.fit_transform(data_new['transmission'])
    data_new['owner']=lb.fit_transform(data_new['owner'])         
       
         
    if st.button('Predict'):
        pred=loaded_model.predict(data_new)
        if pred>0:
            st.balloons()
        message="You can sell your car for {:.2f} lakhs".format(pred[0])
        st.success(message)
    else:
           st.warning("you can't able to sell this car")

        
        
       
   
if __name__ == '__main__':
	main()


