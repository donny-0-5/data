import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('data.csv')


# 페이지 설정
st.set_page_config(
    page_title="Machine Learning Report",
    page_icon="🎉"
)


# 사이드바 및 헤더
st.sidebar.header('머신러닝 보고서')
st.sidebar.write('유방암 진단 데이터로 양성과 악성을 예측한 머신러닝 결과를 요약하고, 추가 분석 가능성을 제시합니다.')

st.header('🎉 Machine Learning Report', divider='rainbow')


# 데이터 로드 및 전처리
df = pd.read_csv('data.csv')
st.header("위스콘신 유방암 머신러닝 보고서")

st.markdown('''
- **Breast Cancer Dataset**: 30개의 입력 기능을 사용하여 선형적으로 분리 가능
- **인스턴스 수**: 569개
- **등급 분포**: 212 악성(M), 357 양성(B)
''')


# 'diagnosis'를 이진 숫자형으로 변환 (M: 1, B: 0)
df["diagnosis"] = df["diagnosis"].apply(lambda x: 1 if x == "M" else 0)

# 불필요한 열 제거
if "Unnamed: 32" in df.columns:
    df.drop("Unnamed: 32", axis=1, inplace=True)


# 데이터 보기
st.subheader('데이터 보기')
st.write(df)


# 데이터 분리 및 표준화
X = df.iloc[:, 1:]  # 특징 데이터
y = df["diagnosis"]  # 레이블

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터셋 분할 (훈련: 80%, 테스트: 20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 로지스틱 회귀 모델 학습
model = LogisticRegression()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)

# 모델 평가 결과 출력
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### 모델 정확도: {accuracy:.2%}")
st.write("### 절대 오차: {:.2f}".format(np.mean(np.abs(y_test - y_pred))))
st.subheader("분류 보고서")
st.text(classification_report(y_test, y_pred))


# 혼동 행렬 시각화
st.subheader("혼동 행렬")
fig = plt.figure(figsize=(10, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Benign (B)", "Malignant (M)"], yticklabels=["Benign (B)", "Malignant (M)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)


# 추가 분석 제안
st.markdown("""
### 추가 분석 제안
- **다른 머신러닝 알고리즘** (e.g., SVM, Random Forest) 적용 및 비교
- **특징 중요도 분석**을 통해 진단에 가장 큰 영향을 미치는 요소 탐구
- 데이터의 **이상치 탐지 및 제거**로 모델 성능 향상 가능성 평가
- **ROC 곡선**을 그려 모델의 분류 성능 시각화
""")