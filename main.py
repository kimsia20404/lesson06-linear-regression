
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "회귀 직선으로 선택한 연도의 예상 기온을 확인해 봅니다."
)


@st.cache_data
def load_data():
    # UTF-8 인코딩으로 CSV 읽기
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 평균기온 정리
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 수업 기준 기간: 2025년까지
    df = df[df["날짜"].dt.year <= 2025].copy()

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    # 연도별 평균기온과 유효 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 관측일이 300일 이상인 해만 사용
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오지 못했습니다.")
    st.info(
        "인터넷 연결과 데이터 URL을 확인한 뒤 다시 실행해 주세요."
    )
    st.stop()

if len(yearly) < 2:
    st.error("회귀 직선을 계산하기에 유효한 연도가 부족합니다.")
    st.stop()

# 회귀 직선 계산
# 독립 변수는 연도 - 1908
x = yearly["연도"].to_numpy(dtype=float) - 1908
y = yearly["연평균기온"].to_numpy(dtype=float)

slope, intercept = np.polyfit(x, y, 1)
correlation = np.corrcoef(x, y)[0, 1]

year_count = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 슬라이더
st.subheader("🔎 예상 기온 알아보기")

selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temp = slope * (selected_year - 1908) + intercept

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label=f"{selected_year}년 예상 연평균기온",
        value=f"{predicted_temp:.2f} °C"
    )

with col2:
    st.metric(
        label="연도와 기온의 상관계수",
        value=f"{correlation:.3f}"
    )

with col3:
    st.metric(
        label="회귀 직선에 사용한 연도 수",
        value=f"{year_count}년"
    )

st.caption(
    f"분석 기간: {start_year}년 ~ {end_year}년 "
    f"({year_count}개 연도, 연간 관측일 300일 이상)"
)

# 그래프용 회귀 직선
# 가로축은 실제 연도를 그대로 사용
line_years = np.arange(start_year, 2101)
line_temps = slope * (line_years - 1908) + intercept

fig = go.Figure()

# 관측된 연평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly["연도"],
        y=yearly["연평균기온"],
        mode="markers",
        name="관측된 연평균기온",
        marker=dict(size=7),
        customdata=yearly[["관측일수"]].to_numpy(),
        hovertemplate=(
            "연도: %{x}년<br>"
            "연평균기온: %{y:.2f} °C<br>"
            "관측일수: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "회귀 추정 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 슬라이더로 선택한 연도의 예상 기온
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상 기온",
        marker=dict(size=14, symbol="star"),
        hovertemplate=(
            "선택 연도: %{x}년<br>"
            "예상 기온: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="범례",
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.update_xaxes(
    range=[1900, 2100],
    dtick=20
)

st.plotly_chart(fig, use_container_width=True)

# 회귀 분석 설명
st.subheader("📊 회귀 분석 결과")

st.write(
    f"- **사용한 연도 수:** {year_count}개"
)
st.write(
    f"- **시작 연도:** {start_year}년"
)
st.write(
    f"- **끝 연도:** {end_year}년"
)
st.write(
    f"- **상관계수:** {correlation:.4f}"
)
st.write(
    f"- **연간 추세:** {slope:.4f} °C/년"
)

if correlation > 0:
    st.write(
        "해석: 연도와 연평균기온 사이에 양의 선형 관계가 나타납니다."
    )
elif correlation < 0:
    st.write(
        "해석: 연도와 연평균기온 사이에 음의 선형 관계가 나타납니다."
    )
else:
    st.write(
        "해석: 연도와 연평균기온 사이의 선형 상관관계가 거의 없습니다."
    )

st.caption(
    "주의: 회귀 직선은 과거 자료의 선형 추세를 연장한 추정치입니다. "
    "미래의 실제 기온은 자연 변동과 기후 변화 등으로 달라질 수 있습니다."
)
