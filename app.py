
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="⚽", layout="centered")

@st.cache_resource
def load():
    return (joblib.load("wc_model.pkl"), joblib.load("team_stats.pkl"),
            pd.read_pickle("fixtures.pkl"), joblib.load("goals_coef.pkl"), joblib.load("groups.pkl"))

model, stats, fixtures, goals_model, groups = load()
FEATS = ["he","ae","elo_diff","h_wr","h_gd","a_wr","a_gd","neutral_i"]
teams = sorted(stats.keys())
elo = {t: stats[t]["elo"] for t in stats}
B0, BE = goals_model["b0"], goals_model["be"]

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"

def _kfac(t):
    if t == "FIFA World Cup": return 60
    if "World Cup" in t or "Confederations" in t: return 50
    if any(x in t for x in ["UEFA Euro","Copa","Africa Cup","Asian Cup","Gold Cup","Nations League"]): return 40
    if "qualification" in t: return 30
    return 20

@st.cache_data(show_spinner=False)
def track_accuracy():
    df = pd.read_csv(RESULTS_URL, parse_dates=["date"])
    pl = df.dropna(subset=["home_score","away_score"]).copy()
    pl["home_score"] = pl["home_score"].astype(int); pl["away_score"] = pl["away_score"].astype(int)
    pl = pl.sort_values("date").reset_index(drop=True)
    el = {}; g = lambda t: el.get(t, 1500); hh = {}; recs = []
    for _, m in pl.iterrows():
        he, ae = g(m.home_team), g(m.away_team); adv = 0 if m.neutral else 100
        hwr = np.mean([x[0] for x in hh.get(m.home_team, [])[-10:]]) if hh.get(m.home_team) else 0.5
        awr = np.mean([x[0] for x in hh.get(m.away_team, [])[-10:]]) if hh.get(m.away_team) else 0.5
        hgd = np.mean([x[1] for x in hh.get(m.home_team, [])[-10:]]) if hh.get(m.home_team) else 0.0
        agd = np.mean([x[1] for x in hh.get(m.away_team, [])[-10:]]) if hh.get(m.away_team) else 0.0
        res = 0 if m.home_score > m.away_score else (2 if m.home_score < m.away_score else 1)
        if m.tournament == "FIFA World Cup" and m.date >= pd.Timestamp("2026-01-01"):
            recs.append((m.date, m.home_team, m.away_team, int(m.home_score), int(m.away_score), res,
                         [he, ae, he-ae, hwr, hgd, awr, agd, int(m.neutral)]))
        exp = 1/(1+10**(-((he+adv)-ae)/400))
        s = 1 if m.home_score > m.away_score else (0 if m.home_score < m.away_score else 0.5)
        k = _kfac(m.tournament) * (1 + np.log(max(abs(m.home_score-m.away_score), 1))*0.5)
        el[m.home_team] = he + k*(s-exp); el[m.away_team] = ae + k*((1-s)-(1-exp))
        hh.setdefault(m.home_team, []).append((s, m.home_score-m.away_score))
        hh.setdefault(m.away_team, []).append((1-s if s != 0.5 else 0.5, m.away_score-m.home_score))
    if not recs:
        return None, None
    X = pd.DataFrame([r[6] for r in recs], columns=FEATS)
    proba = model.predict_proba(X); pick = proba.argmax(1)
    rows = []; correct = 0
    for i, r in enumerate(recs):
        d, h, a, hs, as_, actual, _ = r; pk = pick[i]; correct += (pk == actual)
        pteam = h if pk == 0 else (a if pk == 2 else "Draw")
        ateam = h if actual == 0 else (a if actual == 2 else "Draw")
        rows.append({"Date": str(d)[:10], "Match": f"{h} v {a}", "Score": f"{hs}-{as_}",
                     "Model Pick": pteam, "Actual": ateam, "Conf": f"{proba[i][pk]*100:.0f}%",
                     "Hit": "✅" if pk == actual else "❌"})
    decisive = [(pick[i], recs[i][5]) for i in range(len(recs)) if recs[i][5] != 1]
    dc = sum(1 for pk, a in decisive if pk == a)
    stats = {"n": len(recs), "acc": correct/len(recs),
             "dec_acc": dc/len(decisive) if decisive else 0, "dec_n": len(decisive)}
    return pd.DataFrame(rows), stats


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

tab1, tab2, tab3, tab4 = st.tabs(["Single Match", "Group Fixtures", "Cup Simulator", "Accuracy Tracker"])

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

with tab4:
    st.write("How the model's predictions have held up against actual World Cup results.")
    if st.button("Refresh results", type="primary"):
        track_accuracy.clear()
    table, stats = track_accuracy()
    if stats is None:
        st.info("No completed World Cup 2026 matches in the data yet.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Overall accuracy", f"{stats['acc']*100:.1f}%", f"{stats['n']} matches")
        c2.metric("On decisive games", f"{stats['dec_acc']*100:.1f}%", f"{stats['dec_n']} non-draws")
        st.caption("The model never predicts draws, so group-stage draws count as misses. "
                   "Knockout games always produce a winner, so accuracy should rise in the bracket.")
        st.dataframe(table, use_container_width=True, hide_index=True)
