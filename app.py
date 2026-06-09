
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="centered")

@st.cache_resource
def load():
    return joblib.load("wc_model.pkl"), joblib.load("team_stats.pkl"), pd.read_pickle("fixtures.pkl")

model, stats, fixtures = load()
FEATS = ["he","ae","elo_diff","h_wr","h_gd","a_wr","a_gd","neutral_i"]
teams = sorted(stats.keys())

def predict(home, away, neutral=True):
    H, A = stats[home], stats[away]
    X = pd.DataFrame([[H["elo"], A["elo"], H["elo"]-A["elo"],
                       H["wr"], H["gd"], A["wr"], A["gd"], int(neutral)]], columns=FEATS)
    return model.predict_proba(X)[0]  # [home, draw, away]

st.title("⚽ World Cup 2026 Predictor")
st.caption("XGBoost model on international results since 1872 • Elo + recent form")

tab1, tab2 = st.tabs(["Single Match", "All Group Fixtures"])

with tab1:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Team A", teams, index=teams.index("Spain"))
    away = c2.selectbox("Team B", teams, index=teams.index("England"))
    neutral = st.checkbox("Neutral venue", value=True,
                          help="Uncheck if Team A is the host playing at home")
    if st.button("Predict", type="primary"):
        if home == away:
            st.warning("Pick two different teams.")
        else:
            p = predict(home, away, neutral)
            st.subheader("Result probability")
            st.write(f"**{home} win**"); st.progress(float(p[0]), text=f"{p[0]*100:.0f}%")
            st.write("**Draw**");        st.progress(float(p[1]), text=f"{p[1]*100:.0f}%")
            st.write(f"**{away} win**"); st.progress(float(p[2]), text=f"{p[2]*100:.0f}%")
            st.caption(f"Elo: {home} {stats[home]['elo']:.0f} vs {away} {stats[away]['elo']:.0f}")

with tab2:
    st.write(f"Predicting all {len(fixtures)} group-stage fixtures (neutral venue):")
    out = []
    for _, m in fixtures.iterrows():
        h, a = m.home_team, m.away_team
        if h in stats and a in stats:
            p = predict(h, a, True)
            pick = h if p[0]==max(p) else (a if p[2]==max(p) else "Draw")
            out.append({"Date": str(m.date)[:10], "Match": f"{h} vs {a}",
                        "Team A Win": f"{p[0]*100:.0f}%", "Draw": f"{p[1]*100:.0f}%",
                        "Team B Win": f"{p[2]*100:.0f}%", "Predicted": pick})
    st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)
