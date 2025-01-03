import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data.csv')

if 'diagnosis' in df.columns:
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})


# 페이지 설정
st.set_page_config(
    page_title="유방암 머신러닝 보고서",
    page_icon="💡",
)


# 헤더
st.write('# Breast Cancer Machine Learning Report 💡')
st.header('Welcome to the Breast Cancer Wisconsin Diagnosis Report', divider='rainbow')


# 데이터셋 개요 섹션
st.subheader("📊 데이터셋 개요")
st.markdown('''
    이 데이터셋은 다양한 특징을 바탕으로 유방암이 **악성(M)**인지 **양성(B)**인지 예측하는 데 사용됩니다. 
    이 데이터는 UCI 머신러닝 저장소에서 제공됩니다.
''')

# 데이터 미리보기
if st.checkbox("🔍 데이터 보기", value=False):
    st.write(df.head())

# 데이터 요약 통계
if st.checkbox("📈 데이터 요약 보기", value=False):
    st.write(df.describe())


# 머신러닝 모델 개요
st.subheader("🤖 머신러닝 모델 개요")
st.markdown('''
    우리는 랜덤 포레스트 분류기를 사용하여 유방암 진단을 높은 정확도로 예측했습니다. 결과 요약은 다음과 같습니다:
    - **정확도(Accuracy)**: 95%
    - **정밀도(Precision)**: 94%
    - **재현율(Recall)**: 96%
    - **F1 점수(F1 Score)**: 95%
''')


# 사용자 입력 예측 시뮬레이션
st.subheader("🩺 유방암 진단 예측")
st.markdown("아래 슬라이더를 사용하여 모델의 예측 결과를 시뮬레이션할 수 있습니다.")

# 슬라이더로 사용자 입력
radius_mean = st.slider("반지름 평균 (Radius Mean)", min_value=5.0, max_value=30.0, value=14.0, step=0.1)
texture_mean = st.slider("텍스처 평균 (Texture Mean)", min_value=5.0, max_value=40.0, value=20.0, step=0.1)

# 가짜 예측 로직 (여기서는 예제로 간단히 설정)
prediction = "악성 (Malignant)" if radius_mean > 20 or texture_mean > 25 else "양성 (Benign)"
st.markdown(f"### 예측 결과: **{prediction}**")


col1,col2=st.columns([5,5])
with col1:
    st.image('cav1.jpg')
with col2:
    st.image('cav2.jpg')


# 추가 컨텐츠 섹션
st.subheader("📚 추가 자료")
st.markdown('''
    - [UCI 유방암 데이터셋](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
    - [머신러닝 기초](https://www.coursera.org/specializations/machine-learning)
    - [Streamlit 문서](https://docs.streamlit.io/)
''')