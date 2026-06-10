
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="centered")

@st.cache_resource
def load():
    return (joblib.load("wc_model.pkl"), joblib.load("team_stats.pkl"),
            pd.read_pickle("fixtures.pkl"), joblib.load("goals_model.pkl"), joblib.load("groups.pkl"))

model, stats, fixtures, goals_model, groups = load()
FEATS = ["he","ae","elo_diff","h_wr","h_gd","a_wr","a_gd","neutral_i"]
teams = sorted(stats.keys())
elo = {t: stats[t]["elo"] for t in stats}
B0, BE = goals_model.intercept_, goals_model.coef_[0]

def predict(home, away, neutral=True):
    H, A = stats[home], stats[away]
    X = pd.DataFrame([[H["elo"], A["elo"], H["elo"]-A["elo"],
                       H["wr"], H["gd"], A["wr"], A["gd"], int(neutral)]], columns=FEATS)
    return model.predict_proba(X)[0]

def lam(a, d):
    return np.exp(B0 + BE*(elo[a]-elo[d])/100.0)

def ko_winner(a, b):
    ga, gb = np.random.poisson(lam(a,b)), np.random.poisson(lam(b,a))
    if ga > gb: return a
    if gb > ga: return b
    return a if np.random.random() < 1/(1+10**(-(elo[a]-elo[b])/400)) else b

@st.cache_data(show_spinner=False)
def simulate(n_sims):
    from collections import defaultdict
    tally = defaultdict(lambda: defaultdict(int))
    glist = list(groups.items())
    for _ in range(n_sims):
        standings = {}; thirds = []
        for gl, tms in glist:
            tab = {t: [0,0,0] for t in tms}
            for i in range(4):
                for j in range(i+1, 4):
                    a, b = tms[i], tms[j]
                    ga, gb = np.random.poisson(lam(a,b)), np.random.poisson(lam(b,a))
                    tab[a][2]+=ga; tab[b][2]+=gb; tab[a][1]+=ga-gb; tab[b][1]+=gb-ga
                    if ga>gb: tab[a][0]+=3
                    elif gb>ga: tab[b][0]+=3
                    else: tab[a][0]+=1; tab[b][0]+=1
            rank = sorted(tms, key=lambda t:(tab[t][0],tab[t][1],tab[t][2],np.random.random()), reverse=True)
            standings[gl] = rank
            thirds.append((rank[2], tab[rank[2]][0], tab[rank[2]][1], tab[rank[2]][2]))
        best = [x[0] for x in sorted(thirds, key=lambda x:(x[1],x[2],x[3],np.random.random()), reverse=True)[:8]]
        q = []
        for gl,_ in glist: q += standings[gl][:2]
        q += best
        for t in q: tally[t]["R32"]+=1
        seeded = sorted(q, key=lambda t:-elo[t])
        cur = [ko_winner(seeded[i], seeded[31-i]) for i in range(16)]
        for t in cur: tally[t]["R16"]+=1
        for rname in ["QF","SF","Final","Champion"]:
            cur = [ko_winner(cur[i], cur[i+1]) for i in range(0,len(cur),2)]
            for t in cur: tally[t][rname]+=1
    rows = []
    for t in tally:
        d = tally[t]
        rows.append({"Team": t, "Win Cup": d["Champion"]/n_sims, "Reach Final": d["Final"]/n_sims,
                     "Reach SF": d["SF"]/n_sims, "Reach QF": d["QF"]/n_sims, "Reach R16": d["R16"]/n_sims})
    return pd.DataFrame(rows).sort_values("Win Cup", ascending=False).reset_index(drop=True)

st.title("⚽ World Cup 2026 Predictor")
st.caption("XGBoost match model + Poisson tournament simulator • Elo + recent form, data since 1872")

tab1, tab2, tab3 = st.tabs(["Single Match", "Group Fixtures", "Cup Simulator"])

with tab1:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Team A", teams, index=teams.index("Spain"))
    away = c2.selectbox("Team B", teams, index=teams.index("England"))
    neutral = st.checkbox("Neutral venue", value=True, help="Uncheck if Team A is the host playing at home")
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

with tab3:
    st.write("Simulate the entire tournament (group stage + knockouts) to estimate each team's odds.")
    st.caption("Knockout bracket is Elo-seeded as a proxy for the official bracket.")
    n = st.select_slider("Number of simulations", [1000,2000,5000,10000], value=5000)
    if st.button("Run simulation", type="primary"):
        with st.spinner(f"Running {n:,} simulations..."):
            res = simulate(n)
        show = res.head(24).copy()
        for c in ["Win Cup","Reach Final","Reach SF","Reach QF","Reach R16"]:
            show[c] = (show[c]*100).map(lambda x: f"{x:.1f}%")
        st.dataframe(show, use_container_width=True, hide_index=True)
