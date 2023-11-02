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
    
    df=pd.read_csv("https://drive.google.com/uc?export=download&id=17KxRuOK4uONRZyfYNgjoX62y2yO5KpLY")
    

    lb=LabelEncoder()
    
    
    with open('gb_regressor_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

     

    st.markdown(html_temp,unsafe_allow_html=True)
    
    st.write(' ')
    st.write(' ')
    
   
   
    st.markdown("##### Are you planning to sell your car? then, predict the best selling price of your car\n##### don't wait for while!!!!\n##### The main reason for this huge market is that when you buy a New Car and sale it just another day without any default on it, the price of car reduces by 30%.\n##### There are also many frauds in the market who not only sale wrong but also they could mislead to wrong price.\n##### So, here I used this following dataset to Predict the price of any used car")
    
    
    
    r1 = st.selectbox("Brand name of your car", df['name'].unique())
    
    r2=st.number_input(" The distance travelled by your car in Kilometers",100,500000,step=100)
    
    r3=st.selectbox("what is the fuel type of your car?",df['fuel'].unique())
       
    r4=st.selectbox("Are you Individual or Dealer or Trustmark Dealer?",df['seller_type'].unique())

    r5=st.selectbox("What is the Transmission Type of your car?",df['transmission'].unique())
    
    r6=st.selectbox("Number of Owners of your car Previously had?",df['owner'].unique())
    
    r7=st.slider(" In which year your car was purchased?",1990,2023)
        
    
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


