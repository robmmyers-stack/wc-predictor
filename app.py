
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="World Cup 2026 Predictor", page_icon="\u26bd", layout="centered")

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
FEATS = ["he","ae","elo_diff","h_wr","h_gd","a_wr","a_gd","neutral_i"]

@st.cache_resource
def load_assets():
    return (joblib.load("wc_model.pkl"), joblib.load("goals_coef.pkl"),
            joblib.load("groups.pkl"), joblib.load("team_stats.pkl"))

model, gcoef, groups, static_stats = load_assets()
B0, BE = gcoef["b0"], gcoef["be"]

def _kfac(t):
    if t == "FIFA World Cup": return 60
    if "World Cup" in t or "Confederations" in t: return 50
    if any(x in t for x in ["UEFA Euro","Copa","Africa Cup","Asian Cup","Gold Cup","Nations League"]): return 40
    if "qualification" in t: return 30
    return 20

@st.cache_data(show_spinner="Updating ratings from latest results...")
def build_state():
    df = pd.read_csv(RESULTS_URL, parse_dates=["date"])
    pl = df.dropna(subset=["home_score","away_score"]).copy()
    pl["home_score"] = pl["home_score"].astype(int); pl["away_score"] = pl["away_score"].astype(int)
    pl = pl.sort_values("date").reset_index(drop=True)
    el = {}; g = lambda t: el.get(t, 1500); hh = {}; completed = []
    for _, m in pl.iterrows():
        he, ae = g(m.home_team), g(m.away_team); adv = 0 if m.neutral else 100
        hwr = np.mean([x[0] for x in hh.get(m.home_team, [])[-10:]]) if hh.get(m.home_team) else 0.5
        awr = np.mean([x[0] for x in hh.get(m.away_team, [])[-10:]]) if hh.get(m.away_team) else 0.5
        hgd = np.mean([x[1] for x in hh.get(m.home_team, [])[-10:]]) if hh.get(m.home_team) else 0.0
        agd = np.mean([x[1] for x in hh.get(m.away_team, [])[-10:]]) if hh.get(m.away_team) else 0.0
        res = 0 if m.home_score > m.away_score else (2 if m.home_score < m.away_score else 1)
        if m.tournament == "FIFA World Cup" and m.date >= pd.Timestamp("2026-01-01"):
            completed.append((m.date, m.home_team, m.away_team, int(m.home_score), int(m.away_score), res,
                              [he, ae, he-ae, hwr, hgd, awr, agd, int(m.neutral)]))
        exp = 1/(1+10**(-((he+adv)-ae)/400))
        s = 1 if res == 0 else (0 if res == 2 else 0.5)
        k = _kfac(m.tournament) * (1 + np.log(max(abs(m.home_score-m.away_score), 1))*0.5)
        el[m.home_team] = he + k*(s-exp); el[m.away_team] = ae + k*((1-s)-(1-exp))
        hh.setdefault(m.home_team, []).append((s, m.home_score-m.away_score))
        hh.setdefault(m.away_team, []).append((1-s if s != 0.5 else 0.5, m.away_score-m.home_score))

    def snap(t):
        h = hh.get(t, [])
        return {"elo": g(t),
                "wr": np.mean([x[0] for x in h[-10:]]) if h else 0.5,
                "gd": np.mean([x[1] for x in h[-10:]]) if h else 0.0}
    snapshot = {t: snap(t) for t in (set(pl.home_team) | set(pl.away_team))}

    up = df[(df.tournament == "FIFA World Cup") & (df.home_score.isna()) & (df.date >= pd.Timestamp("2026-01-01"))]
    up_rows = []
    for _, m in up.iterrows():
        h, a = m.home_team, m.away_team
        if h in snapshot and a in snapshot:
            H, A = snapshot[h], snapshot[a]
            X = pd.DataFrame([[H["elo"], A["elo"], H["elo"]-A["elo"], H["wr"], H["gd"], A["wr"], A["gd"], int(m.neutral)]], columns=FEATS)
            p = model.predict_proba(X)[0]
            fav = h if p[0] >= p[2] else a
            up_rows.append({"Date": str(m.date)[:10], "Match": f"{h} v {a}",
                            "Lean": fav, f"_h": h, "_a": a,
                            "P_home": p[0], "P_draw": p[1], "P_away": p[2]})

    acc_rows = []; correct = 0
    if completed:
        Xc = pd.DataFrame([c[6] for c in completed], columns=FEATS)
        pc = model.predict_proba(Xc); pk = pc.argmax(1)
        for i, c in enumerate(completed):
            d, h, a, hs, as_, actual, _ = c; correct += (pk[i] == actual)
            pteam = h if pk[i] == 0 else (a if pk[i] == 2 else "Draw")
            ateam = h if actual == 0 else (a if actual == 2 else "Draw")
            acc_rows.append({"Date": str(d)[:10], "Match": f"{h} v {a}", "Score": f"{hs}-{as_}",
                             "Model Pick": pteam, "Actual": ateam, "Conf": f"{pc[i][pk[i]]*100:.0f}%",
                             "Hit": "\u2705" if pk[i] == actual else "\u274c"})
        dec = [(pk[i], completed[i][5]) for i in range(len(completed)) if completed[i][5] != 1]
        dc = sum(1 for p, a in dec if p == a)
        acc_stats = {"n": len(completed), "acc": correct/len(completed),
                     "dec_acc": dc/len(dec) if dec else 0, "dec_n": len(dec)}
    else:
        acc_stats = None

    return snapshot, pd.DataFrame(up_rows), pd.DataFrame(acc_rows), acc_stats

snapshot, upcoming_df, acc_df, acc_stats = build_state()
teams = sorted(snapshot.keys())
elo = {t: snapshot[t]["elo"] for t in snapshot}

def predict(home, away, neutral=True):
    H, A = snapshot[home], snapshot[away]
    X = pd.DataFrame([[H["elo"], A["elo"], H["elo"]-A["elo"], H["wr"], H["gd"], A["wr"], A["gd"], int(neutral)]], columns=FEATS)
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
    tally = defaultdict(lambda: defaultdict(int)); glist = list(groups.items())
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

st.title("\u26bd World Cup 2026 Predictor")
st.caption("XGBoost match model + Poisson simulator \u2022 Elo auto-updates from latest results")

tab1, tab2, tab3, tab4 = st.tabs(["Single Match", "Upcoming Matches", "Cup Simulator", "Accuracy Tracker"])

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
            st.caption(f"Elo: {home} {snapshot[home]['elo']:.0f} vs {away} {snapshot[away]['elo']:.0f}")

with tab2:
    st.write("Model predictions for the next World Cup matches still to be played.")
    if st.button("Refresh", key="r2"):
        build_state.clear(); st.rerun()
    if upcoming_df.empty:
        st.info("No upcoming World Cup matches scheduled in the data yet.")
    else:
        disp = []
        for _, r in upcoming_df.iterrows():
            disp.append({"Date": r["Date"], "Match": r["Match"], "Model Lean": r["Lean"],
                         "Team A Win": f"{r['P_home']*100:.0f}%", "Draw (90')": f"{r['P_draw']*100:.0f}%",
                         "Team B Win": f"{r['P_away']*100:.0f}%"})
        st.dataframe(pd.DataFrame(disp), use_container_width=True, hide_index=True)
        st.caption("Team A / Team B follow the order shown in Match. Draw (90') means tied after regulation "
                   "(extra time / penalties decide in knockouts).")

with tab3:
    st.write("Simulate the tournament from the group stage to estimate each team's title odds.")
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
    st.write("How the model's predictions have held up against actual results.")
    if st.button("Refresh results", key="r4"): build_state.clear(); st.rerun()
    if acc_stats is None:
        st.info("No completed World Cup 2026 matches in the data yet.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Overall accuracy", f"{acc_stats['acc']*100:.1f}%", f"{acc_stats['n']} matches")
        c2.metric("On decisive games", f"{acc_stats['dec_acc']*100:.1f}%", f"{acc_stats['dec_n']} non-draws")
        st.caption("The model never predicts draws, so group-stage draws count as misses. "
                   "Knockout games always produce a winner, so accuracy should rise in the bracket.")
        st.dataframe(acc_df, use_container_width=True, hide_index=True)
