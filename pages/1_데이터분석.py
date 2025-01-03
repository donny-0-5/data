import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data.csv')

# 페이지 설정
st.set_page_config(
    page_title='데이터 분석',
    page_icon='💡',
)

# 사이드바 및 헤더
st.sidebar.header('데이터 분석')
st.sidebar.write('유방암 진단 데이터의 주요 통계를 탐구하고, 열 별로 세부 데이터를 분석할 수 있는 도구를 제공합니다.')

st.header('📊 Data Insights', divider='rainbow')

# 데이터 분석 개요 설명
st.markdown('''
    이 페이지는 유방암 진단 데이터를 분석하여 양성과 악성의 주요 특징을 이해하고, 관련된 통계를 직관적으로 제공합니다.
    - 데이터를 탐색하고, 주요 통계를 확인하며, 열별로 세부 정보를 필터링하여 유방암 진단과 관련된 정보를 심층적으로 분석해 보세요.
''')


# 탭 생성
t1, t2, t3, t4 = st.tabs(['상위 데이터', '데이터 통계', '컬럼 선택', '조건 필터링'])


# 탭 1: 상위 데이터
with t1:
    st.subheader("📋 상위 데이터")
    st.write("데이터프레임의 상위 10개 데이터를 표시합니다.")
    dh = df.head(10)
    st.write(dh)


# 탭 2: 데이터 통계
with t2:
    st.subheader("📈 데이터 통계")
    st.write("데이터의 기초 통계량을 확인할 수 있습니다.")
    dd = df.describe()
    st.write(dd)

    # 데이터 통계 시각화
    st.write("📊 **특징별 히스토그램**")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_col = st.selectbox("히스토그램을 표시할 열을 선택하세요", numeric_columns)
    if selected_col:
        fig, ax = plt.subplots()
        sns.histplot(df[selected_col], kde=True, bins=20, ax=ax)
        st.pyplot(fig)


# 탭 3: 컬럼 선택
with t3:
    st.subheader("🔍 컬럼 데이터 선택")
    st.write("분석에 적합한 주요 컬럼들을 선택하여 데이터를 확인하세요.")
    
    # 권장 컬럼 목록
    recommended_columns = [
        'diagnosis', 'radius_mean', 'texture_mean', 'area_mean', 
        'smoothness_mean', 'compactness_mean', 'concavity_mean', 
        'radius_worst', 'texture_worst', 'area_worst'
    ]
    
    col = df.columns.tolist()
    scol = st.multiselect('표시할 컬럼을 선택하세요', col, default=recommended_columns)
    if scol:
        ldf = df.loc[:, scol]
        st.write(ldf)
    else:
        st.write("선택된 컬럼이 없습니다.")


# 탭 4: 조건 필터링
with t4:
    st.subheader("🔧 조건 기반 데이터 필터링")
    st.write("특정 열에서 조건을 지정해 데이터를 필터링합니다.")
    
    # 조건 필터링 UI
    col_options = df.columns.tolist()
    selected_col = st.selectbox("조건을 적용할 컬럼을 선택하세요", col_options)
    
    if selected_col:
        unique_values = df[selected_col].dropna().unique()
        filter_value = st.selectbox(f"'{selected_col}'의 값을 선택하세요", unique_values)
        filtered_df = df[df[selected_col] == filter_value]
        st.write(filtered_df)
    else:
        st.write("조건 필터링을 위한 컬럼을 선택하세요.")