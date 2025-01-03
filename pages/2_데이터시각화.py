import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data.csv')

if 'diagnosis' in df.columns:
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 페이지 설정
st.set_page_config(
    page_title='데이터 시각화',
    page_icon='📊',
    layout="wide",
)

# 사이드바 및 헤더
st.sidebar.header('데이터 시각화')
st.sidebar.write('유방암 진단 데이터를 시각화하여 주요 특징과 진단 결과 간의 관계를 탐색합니다. 그래프를 통해 양성과 악성의 분포, 변수 간 상관관계, 이상치 등을 확인할 수 있습니다.')

st.header('📊 Data Visualization', divider='rainbow')

# 시각화 개요 설명
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('''
데이터 시각화를 통해 유방암 진단과 관련된 다음 질문에 대한 답을 확인할 수 있습니다:
- 양성(B)과 악성(M) 진단에 따라 주요 특징들이 어떻게 달라지나요?
- 암 진단과 관련된 변수 간의 상관 관계는 무엇인가요?
- 데이터의 분포를 통해 유방암 진단에 유용한 통찰을 얻을 수 있나요?
- 이상치나 특이한 분포를 통해 추가적인 정보를 확인할 수 있나요?
''')
st.markdown('</div>', unsafe_allow_html=True)


# 탭 생성
st.markdown('<div class="section">', unsafe_allow_html=True)
t1, t2, t3, t4 = st.tabs(['진단 결과 분포', '특징 분포 비교', '진단별 상관성', '이상치 탐색'])


# 탭 1: 진단 결과 분포
with t1:
    st.header('진단 결과 분포')
    st.write("진단 결과(양성/악성)에 따른 데이터 분포를 확인합니다.")
    fig = plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='diagnosis', palette='coolwarm')
    plt.title('Diagnosis Distribution', fontsize=16)
    plt.xlabel('Diagnosis (B: Benign, M: Malignant)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    st.pyplot(fig)


# 탭 2: 특징 분포 비교
with t2:
    st.header('특징 분포 비교')
    st.write("특정 수치형 특징의 분포를 진단 결과에 따라 비교합니다.")
    selected_feature = st.selectbox(
        "분포를 확인할 특징을 선택하세요",
        ['radius_mean', 'texture_mean', 'area_mean']
    )
    fig = plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=selected_feature, hue='diagnosis', kde=True, palette='coolwarm')
    plt.title(f'Distribution of {selected_feature} by Diagnosis', fontsize=16)
    plt.xlabel(selected_feature, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    st.pyplot(fig)


# 탭 3: 진단별 상관성 (페어플롯)
with t3:
    st.header('특징 간 상관성 분석')
    st.write("선택한 주요 변수들 간의 모든 상관 관계를 확인합니다.")
    
    # 주요 변수 선택
    selected_features = st.multiselect(
        "페어플롯에 포함할 변수를 선택하세요 (최대 5개)",
        ['radius_mean', 'texture_mean', 'area_mean', 'smoothness_mean', 'compactness_mean'],
        default=['radius_mean', 'texture_mean', 'area_mean']
    )
    
    if len(selected_features) > 1:
        fig = sns.pairplot(df, vars=selected_features, hue='diagnosis', palette='coolwarm')
        st.pyplot(fig)
    else:
        st.write("2개 이상의 변수를 선택해야 페어플롯을 생성할 수 있습니다.")


# 탭 4: 이상치 탐색
with t4:
    st.header('이상치 탐색')
    st.write("박스플롯을 통해 특징별 이상치를 탐색합니다.")
    selected_feature = st.selectbox(
        "이상치를 탐색할 특징을 선택하세요",
        ['radius_mean', 'texture_mean', 'area_mean']
    )
    fig = plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, y=selected_feature, x='diagnosis', palette='coolwarm')
    plt.title(f'Boxplot of {selected_feature} by Diagnosis', fontsize=16)
    plt.xlabel('Diagnosis', fontsize=12)
    plt.ylabel(selected_feature, fontsize=12)
    st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)


# CSS 스타일 정의
st.markdown(
    """
    <style>
    /* 페이지 전체 레이아웃 설정 */
    .main {
        max-width: 80%;
        margin: 0 auto;
        padding: 20px;
    }

    /* 섹션 간 마진 추가 */
    .section {
        margin-bottom: 40px;
    }

    /* 페이지 전체의 스크롤바 숨기기 */
    ::-webkit-scrollbar {
        display: none;
    }
    body {
        overflow: -moz-scrollbars-none; /* Firefox용 */
        -ms-overflow-style: none; /* IE 및 Edge용 */
        scrollbar-width: none; /* 최신 브라우저용 */
    }
    </style>
    """,
    unsafe_allow_html=True
)