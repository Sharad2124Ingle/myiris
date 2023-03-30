
import streamlit as st
import pickle
import numpy as np
#import sklearn
model=pickle.load(open('randon_for.pkl','rb')) 

def predict_forest(PL,PW,SL):
    input=np.array([[PL,PW,SL]]).astype(np.float64)
    prediction=model.predict_proba(input)
    pred='{0:.{1}f}'.format(prediction[0][0], 2)
    return float(pred)

def main():
    st.title("IRIS BY SHARAD")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">iris prediction ML App </h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    PL = st.text_input("Petal_length","")
    PW = st.text_input("Petal_width","")
    SL = st.text_input("Sepal_lengthe","")
    
    zero_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> iris setosa</h2>
       </div>
    """
    one_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> iris verginicia</h2>
       </div>
    """
    two_html="""  
      <div style="background-color:#F4D03F;padding:10px >
       <h2 style="color:white;text-align:center;"> iris vernicular</h2>
       </div>
    """
    
    if st.button("Predict"):
        output=predict_forest(PL,PW,SL)
       
        
        if output == 1:
            st.markdown(zero_html,unsafe_allow_html=True)
        if output == 2:
            st.markdown(one_html,unsafe_allow_html=True)
        else:
            st.markdown(two_html,unsafe_allow_html=True)
        
        
        
        
if __name__=='__main__':
  
    main()
    
    

