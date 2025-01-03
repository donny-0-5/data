import pandas as pd
import seaborn as sns 
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report , confusion_matrix
from sklearn.linear_model import LogisticRegression


st.set_page_config(page_title="machine", page_icon="🎉")
st.sidebar.header("breast 머신러닝 보고서")



df = pd.read_csv('data.csv')
st.header("breast cancer 위스콘신 유방암 머신러닝 보고서")

st.markdown('''
- 암 진단이 양성인지 악성인지 여러 관찰/특징에 기초하여 예측 
- 30가지 기능이 사용되며, 예:

        - 반지름(둘레의 중심에서 점까지의 거리 mean)
        - 텍스처(회색 스케일 값의 표준 편차)
        - 둘레의
        - 면적
        - 평활도(반지름 길이의 국부적 변화)
        - 콤팩트성 (perimeter^2 / 면적 - 1.0)
        - 오목한 부분(윤곽의 오목한 부분의 severity)
        - 오목한 점(윤곽의 오목한 부분의 수)
        - 대칭성 
        - 프랙탈 차원("coastline 근사" - 1)

​- 데이터 셋은 30개의 모든 입력 기능을 사용하여 선형적으로 분리 가능
- 인스턴스 수: 569개
- 등급 분포: 212 악성, 357 양성
                   
            ''')


df["diagnosis"] = [1 if value == "M" else 0 for value in df["diagnosis"]]
df.drop("Unnamed: 32", axis = 1, inplace = True)
df["diagnosis"] = df["diagnosis"].astype("category", copy = False)

st.subheader('데이터보기')
st.write(df)


# 데이터 시각화
 
fig = plt.figure(figsize=(10,5)) 

# sns.pairplot(df, hue = 'diagnosis', vars = ['radius_mean', 'texture_mean', 'area_mean'] )

sns.scatterplot(x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis', data = df)

# sns.heatmap(df.corr(), annot=True)
st.pyplot(fig)



# Modeling

x = df.iloc[:, 1:]
y = df["diagnosis"]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# 학습하기

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 42)
model = LogisticRegression()  
model.fit(x_train, y_train)


# 로지스틱 회귀로 학습, 예측, 평가 
y_pred = model.predict(x_test)


# 정확도
#Evaluation of model
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report

accuracy = accuracy_score(y_test, y_pred)
error = mean_absolute_error(y_test, y_pred)

st.write("accuracy is : ", accuracy, " and the absolute error is: ", error)

# Rapport de classification
st.write(classification_report(y_test, y_pred))


# 모델평가

fig=plt.figure(figsize=(15,8))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True)

st.pyplot(fig)