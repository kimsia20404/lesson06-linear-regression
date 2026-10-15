
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

st.title("🌡️ 기온 예측기")
st.write(
    "서울의 과거 연평균기온을 분석하고, "
    "회귀 직선을 이용해 선택한 연도의 예상 기온을 확인해 봅니다."
)

st.info(
    "분석 기준: 2025년까지의 데이터 중 "
    "평균기온이 유효한 관측일이 300일 이상인 연도"
)

# --------------------------------------------------
# 2. 데이터 불러오기 및 전처리
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required_columns = [
        "날짜", "지점", "평균기온", "최저기온", "최고기온"
    ]

    missing = [
        col for col in required_columns if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"필요한 열이 없습니다: {', '.join(missing)}"
        )

    df["날짜"] = pd.to_datetime(
        df["날짜"], errors="coerce"
    )

    df["평균기온"] = pd.to_numeric(
        df["평균기온"], errors="coerce"
    )

    df = df.dropna(
        subset=["날짜", "평균기온"]
    ).copy()

    # 수업 기준 기간: 2025년까지
    df = df[df["날짜"].dt.year <= 2025].copy()

    # 연도별 평균기온과 유효 관측일 수 계산
    df["연도"] = df["날짜"].dt.year

    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("평균기온", "count")
        )
        .reset_index()
    )

    # 유효 관측일이 300일 이상인 연도만 유지
    yearly = yearly[
        yearly["관측일수"] >= 300
    ].copy()

    yearly = yearly.sort_values(
        "연도"
    ).reset_index(drop=True)

    return yearly


try:
    yearly = load_data()
except Exception as e:
    st.error("기온 데이터를 불러오는 중 오류가 발생했습니다.")
    st.caption(f"오류 내용: {e}")
    st.info(
        "인터넷 연결과 데이터 주소를 확인한 뒤 다시 실행해 주세요."
    )
    st.stop()

if len(yearly) < 2:
    st.error(
        "회귀 직선을 계산할 수 있는 연도가 부족합니다. "
        "데이터와 관측일 기준을 확인해 주세요."
    )
    st.stop()

# --------------------------------------------------
# 3. 전체 기간 회귀 분석
# --------------------------------------------------
years = yearly["연도"].to_numpy(dtype=float)
temps = yearly["연평균기온"].to_numpy(dtype=float)

# y = slope * 연도 + intercept
slope, intercept = np.polyfit(years, temps, 1)

correlation = np.corrcoef(years, temps)[0, 1]

year_count = len(yearly)
start_year = int(yearly["연도"].min())
end_year = int(yearly["연도"].max())

# 100년당 기온 변화량
rise_per_100_years = slope * 100

# --------------------------------------------------
# 4. 최근 20년 회귀 분석
# --------------------------------------------------
# 마지막 관측 연도를 포함해 20년 범위로 설정
recent_start = end_year - 19

recent = yearly[
    yearly["연도"] >= recent_start
].copy()

recent_slope = np.nan
recent_intercept = np.nan
recent_correlation = np.nan
recent_rise_per_100_years = np.nan

if len(recent) >= 2:
    recent_years = recent["연도"].to_numpy(dtype=float)
    recent_temps = recent["연평균기온"].to_numpy(dtype=float)

    recent_slope, recent_intercept = np.polyfit(
        recent_years, recent_temps, 1
    )

    recent_correlation = np.corrcoef(
        recent_years, recent_temps
    )[0, 1]

    recent_rise_per_100_years = recent_slope * 100

# --------------------------------------------------
# 5. 연도 슬라이더와 예상 기온
# --------------------------------------------------
st.subheader("🔎 몇 년도의 기온이 궁금한가요?")

selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=min(max(end_year, 1900), 2100),
    step=1
)

predicted_temp = (
    slope * selected_year + intercept
)

st.metric(
    label=f"{selected_year}년 예상 연평균기온",
    value=f"{predicted_temp:.2f} °C"
)

st.caption(
    "과거 전체 기간의 회귀 직선을 선택한 연도까지 연장한 값입니다. "
    "실제 미래 기온을 보장하는 예보는 아닙니다."
)

# --------------------------------------------------
# 6. 100년당 기온 변화량 비교
# --------------------------------------------------
st.subheader("🌡️ 100년당 기온 변화량 비교")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year})",
        value=f"{rise_per_100_years:+.2f} °C",
        help=(
            "전체 분석 기간의 회귀 직선 기울기를 "
            "100배 한 값입니다."
        )
    )

with col2:
    if len(recent) >= 2:
        actual_recent_start = int(recent["연도"].min())
        actual_recent_end = int(recent["연도"].max())

        st.metric(
            label=(
                f"최근 20년 "
                f"({actual_recent_start}~{actual_recent_end})"
            ),
            value=f"{recent_rise_per_100_years:+.2f} °C",
            delta=(
                f"{recent_rise_per_100_years - rise_per_100_years:+.2f} °C"
                " (전체 기간 대비)"
            ),
            help=(
                "최근 20년 자료로 계산한 기울기를 "
                "100년당 변화량으로 환산했습니다."
            )
        )
    else:
        st.metric(
            label="최근 20년",
            value="계산 불가"
        )

st.caption(
    "양수(+)는 상승 추세, 음수(-)는 하강 추세입니다. "
    "최근 20년의 값을 100년 기준으로 환산한 것은 비교를 위한 것이며, "
    "실제로 100년 동안 관측했다는 뜻은 아닙니다."
)

# --------------------------------------------------
# 7. Plotly 산점도 및 회귀 직선
# --------------------------------------------------
st.subheader("📈 서울 연평균기온과 회귀 직선")

fig = go.Figure()

# 관측된 연평균기온
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
            "유효 관측일: %{customdata[0]}일"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀 직선
line_years = np.arange(start_year, 2101)
line_temps = slope * line_years + intercept

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temps,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(width=3),
        hovertemplate=(
            "연도: %{x}년<br>"
            "전체 기간 추세: %{y:.2f} °C"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선
if len(recent) >= 2:
    recent_line_years = np.arange(
        int(recent["연도"].min()),
        int(recent["연도"].max()) + 1
    )

    recent_line_temps = (
        recent_slope * recent_line_years + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temps,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(width=3, dash="dash"),
            hovertemplate=(
                "연도: %{x}년<br>"
                "최근 20년 추세: %{y:.2f} °C"
                "<extra></extra>"
            )
        )
    )

# 선택한 연도의 예상 기온
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
    title="서울 연평균기온: 관측값과 장기 추세",
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    hovermode="closest",
    legend_title="그래프 항목",
    margin=dict(l=20, r=20, t=60, b=20)
)

fig.update_xaxes(
    range=[1900, 2100],
    dtick=20
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# 8. 분석 기간과 통계 정보
# --------------------------------------------------
st.subheader("📊 분석에 사용한 데이터")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        label="사용한 연도 수",
        value=f"{year_count}개"
    )

with info2:
    st.metric(
        label="시작 연도",
        value=f"{start_year}년"
    )

with info3:
    st.metric(
        label="끝 연도",
        value=f"{end_year}년"
    )

st.write(
    f"**전체 기간 상관계수:** {correlation:.4f}"
)

st.write(
    f"**전체 기간 연간 기울기:** {slope:+.4f} °C/년"
)

if len(recent) >= 2:
    st.write(
        f"**최근 20년 사용 연도 수:** {len(recent)}개"
    )
    st.write(
        f"**최근 20년 상관계수:** {recent_correlation:.4f}"
    )
    st.write(
        f"**최근 20년 연간 기울기:** "
        f"{recent_slope:+.4f} °C/년"
    )

st.subheader("💡 결과 해석하기")

if rise_per_100_years > 0:
    st.write(
        f"전체 기간의 선형 추세에서는 100년당 "
        f"약 {rise_per_100_years:.2f} °C 상승하는 경향이 나타납니다."
    )
elif rise_per_100_years < 0:
    st.write(
        f"전체 기간의 선형 추세에서는 100년당 "
        f"약 {abs(rise_per_100_years):.2f} °C 하강하는 경향이 나타납니다."
    )
else:
    st.write(
        "전체 기간의 선형 추세에서 기울기는 0에 가깝습니다."
    )

if len(recent) >= 2:
    difference = (
        recent_rise_per_100_years - rise_per_100_years
    )

    if difference > 0:
        st.write(
            "최근 20년의 기울기가 전체 기간보다 큽니다. "
            "두 기간의 선형 추세를 비교하면 최근의 상승 경향이 더 큽니다."
        )
    elif difference < 0:
        st.write(
            "최근 20년의 기울기가 전체 기간보다 작습니다. "
            "두 기간의 선형 추세를 비교하면 최근의 상승 경향이 더 작거나 "
            "하강 경향이 더 큽니다."
        )
    else:
        st.write(
            "최근 20년과 전체 기간의 기울기가 거의 같습니다."
        )

st.caption(
    "주의: 상관관계와 회귀 직선은 자료의 통계적 관계를 보여 줍니다. "
    "기후의 모든 원인을 설명하거나 미래의 실제 기온을 확정하지는 않습니다."
)
