
import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image

model_predictor=pickle.load(open('logistic.sav','rb'))
colonnes=['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
       'ARPU_SEGMENT', 'FREQUENCE', 'ON_NET', 'ORANGE', 'MRG', 'REGULARITY',
       'TOP_PACK', 'FREQ_TOP_PACK']
#data_input=np.array(['SAINT-LOUIS','H 15-18 month',500.0,1.0,1000.0,333.0,1.0,7.0,6.0,'NO',7,'All-net 500F=2000F;5d',1.0]).reshape(1,-1)



def recup_data_client(data_input):
       data_input_trans=np.array(data_input).reshape(1,-1)

       tab_inputs=pd.DataFrame(data_input_trans, columns=colonnes)
       y_predit=model_predictor.predict(tab_inputs)
       #print(y_predit[0])
       if (y_predit[0]==0):
              return "Le Client Ne se désabonne Pas"
       else:
              return "Le Client Va se Désabonner"

def main():
       imageExpres=Image.open('imExpresso.jpg')
       st.image(imageExpres,width=350)
       st.title("Application de Prédiction de Fidélité de Clients (Expresso)")
       infos=[]
       for col in colonnes:
              info=st.text_input("{}".format(col))
              infos.append(info)
       prediction=""
       if st.button("Cliquer pour vour le Résultat"):
              prediction=recup_data_client(infos)
       st.success(prediction)


if __name__=='__main__':
       main() 

