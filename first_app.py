import streamlit as st
import pandas as pd
import numpy as np
from PIL  import Image
from pycaret.classification import load_model,predict_model
modelo=load_model('Stroke_model_1')


#Funcion de prediccion de base de datos
def predicciones(modelo,base_datos):
    prediction_df=predict_model(estimator=modelo,data=base_datos)
    predicciones=prediction_df['Label'][0]
    
    return predicciones

#diseño aplicacion
def app():
    imagen=Image.open("images.jpg")
    
    st.image(imagen,caption='stroke prediction model',use_column_width=True)
    
    select_box=st.sidebar.selectbox('Tipo de prediccion',{'online'})
    st.sidebar.info('Stroke prediccions base on Age,hypertension')
    
    #st.title('stroke prediction model')
    st.markdown("<h1 style='text-align: center; color: red;'>stroke prediction model</h1>", unsafe_allow_html=True)
    
    if select_box=='online':
       hypertension= st.selectbox('hypertension',[0,1])
       heart_disease=st.selectbox('heart_disease',[0,1])
       avg_glucose_level=st.number_input('average glucose level',min_value=55.12,max_value=271.74,value=106.14)
       age=st.number_input('patience Age',min_value=1,value=43)
     
       output=""
       
       input_dict={'hypertension':hypertension,'heart_disease':heart_disease,
                   'avg_glucose_level':avg_glucose_level,
                   'age':age}
       
       base_datos=pd.DataFrame([input_dict])
      
       if st.button('Make a prediction'):
           output=predicciones(modelo=modelo,base_datos=base_datos)
           if output==1:
               output='Posible stroke'
           else:
               output='No risk to suffer a stroke'
               
           
       
           st.success('la prediccion es :{}'.format(output))

if __name__=='__main__':
    app()
