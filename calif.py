import numpy as np
import streamlit as st
import pandas as pd

st.write(''' # Predicción de calificación  ''')
st.image("calis.jpg", caption="Predice de tu calificación.")

st.header('Datos')

def user_input_features():
  # Entrada
  hours_studied = st.number_input('hours_studied:', min_value=0, max_value=15, value = 0)
  sleep_hours = st.number_input('sleep_hours',  min_value=0, max_value=15, value = 0)
  attendance_percent = st.number_input('attendance_percent', min_value=0, max_value=100, value = 0)
  exam_score = st.number_input('exam_score', min_value=0, max_value=50, value = 0)


  user_input_data = {'hours_studied': Horas de estudio,
                     'sleep_hours': Horas de sueño,
                     'attendance_percent': Porcentaje de asistencia,
                     'previous_scores': calificaciones previas
                     }

  features = pd.DataFrame(user_input_data, index=[0])

  return features

df = user_input_features()

datos =  pd.read_csv('student_exam_scores.csv', encoding='latin-1')
X = datos.drop(columns='student_id', 'exam_score')
y = datos['exam_score']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1613318)
LR = LinearRegression()
LR.fit(X_train,y_train)

b1 = LR.coef_
b0 = LR.intercept_
prediccion = b0 + b1[0]*df['hours_studied'] + b1[1]*df['sleep_hours'] + b1[2]*df['attendance_percent'] + b1[3]*df[previous_scores']
st.subheader('Cálculo de calificación')
st.write('' es ', prediccion)
