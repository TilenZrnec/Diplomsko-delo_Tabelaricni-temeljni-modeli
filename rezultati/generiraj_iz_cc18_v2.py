"""Tabele in slike za poglavje Rezultati, iz datotek zagona cc18_v2.

Zagon (iz korena repozitorija s kodo, v okolju s pandas in matplotlib):
    cd ../Diplomsko-delo_Koda
    python ../Diplomsko-delo_Tabelaricni-temeljni-modeli/rezultati/generiraj_iz_cc18_v2.py \
        ../Diplomsko-delo_Tabelaricni-temeljni-modeli/rezultati \
        ../Diplomsko-delo_Tabelaricni-temeljni-modeli/slike
Na koncu izpiše še številke, ki so povzete v besedilu poglavja.
"""
import re, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.stats import cd_diagram
R = 'results/runs/cc18_v2'; S = R + '/summary'
OUT = sys.argv[1]; SLIKE = sys.argv[2]
IME = {'random_forest': 'Naključni gozd', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
       'catboost': 'CatBoost', 'tabpfn': 'TabPFN', 'tabicl': 'TabICL'}
ov = pd.read_csv(S + '/overall.csv'); ovn = pd.read_csv(S + '/overall_nonsaturated.csv')
ORDER = list(ov.algorithm)  # po povprečnem rangu
def n(x, d=4): return f"\\num{{{x:.{d}f}}}"
def e(p):
    return f"\\num{{{p:.1e}}}" if p < 1e-3 else f"\\num{{{p:.4f}}}"
def src(files): return "% Samodejno iz Diplomsko-delo_Koda/" + ", ".join(files) + " (zagon cc18_v2, commit dc1aeb2). Ne urejaj na roko.\n"

# 1. Skupna primerjava
m = ovn.set_index('algorithm')
t = src([S + '/overall.csv', S + '/overall_nonsaturated.csv'])
t += r"""\begin{table}[htbp]
\centering
\caption{Skupna primerjava algoritmov na zbirki OpenML-CC18. Povprečni ROC-AUC je izračunan le na podatkovnih množicah, na katerih so uspeli vsi algoritmi (71 od 72 oziroma 38 od 39), povprečni rang pa na vseh; neuspeh na podatkovni množici prejme najslabši rang. Zmage štejejo prvo mesto, vključno z izenačitvami.}
\label{tab:skupno}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\hline
 & \multicolumn{4}{c}{vseh 72 množic} & \multicolumn{3}{c}{39 nenasičenih množic} \\
Algoritem & ROC-AUC & rang & zmage & neuspehi & ROC-AUC & rang & zmage \\
\hline
"""
for _, r in ov.iterrows():
    q = m.loc[r.algorithm]
    t += f"{IME[r.algorithm]} & {n(r.mean_roc_auc_complete)} & {n(r.mean_rank,2)} & {int(r.n_wins)} & {int(r.n_failed_datasets)} & {n(q.mean_roc_auc_complete)} & {n(q.mean_rank,2)} & {int(q.n_wins)} \\\\\n"
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
open(OUT + '/tab_skupno.tex', 'w').write(t)

# 2. Wilcoxon + Holm
def wil(f):
    w = pd.read_csv(f); rows = {}
    for _, r in w.iterrows():
        a, b, d, wa, wb = r.a, r.b, r.mean_diff, r.wins_a, r.wins_b
        if ORDER.index(a) > ORDER.index(b): a, b, d, wa, wb = b, a, -d, wb, wa
        rows[(a, b)] = (d, wa, wb, r.p_holm, r.significant)
    return rows
wa_, wn_ = wil(S + '/wilcoxon_holm.csv'), wil(S + '/wilcoxon_holm_nonsaturated.csv')
keys = sorted(wa_, key=lambda k: (ORDER.index(k[0]), ORDER.index(k[1])))
t = src([S + '/wilcoxon_holm.csv', S + '/wilcoxon_holm_nonsaturated.csv'])
t += r"""\begin{table}[htbp]
\centering
\caption{Parni Wilcoxonovi testi predznačenih rangov s Holmovim popravkom. Razlika je povprečna razlika ROC-AUC (prvi minus drugi algoritem) čez podatkovne množice, na katerih sta uspela oba; zmage so število množic, na katerih je boljši prvi oziroma drugi. Z zvezdico so označene razlike, značilne pri $\alpha = 0{,}05$.}
\label{tab:wilcoxon}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrlrrl}
\hline
 & \multicolumn{3}{c}{vseh 72 množic} & \multicolumn{3}{c}{39 nenasičenih množic} \\
Par algoritmov & razlika & zmage & $p_\mathrm{Holm}$ & razlika & zmage & $p_\mathrm{Holm}$ \\
\hline
"""
for k in keys:
    row = f"{IME[k[0]]} -- {IME[k[1]]}"
    for W in (wa_, wn_):
        d, x, y, p, s = W[k]
        row += f" & {n(d)} & {int(x)}:{int(y)} & {e(p)}{'$^{*}$' if s else ''}"
    t += row + " \\\\\n"
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
open(OUT + '/tab_wilcoxon.tex', 'w').write(t)

# 3. Časi
df = pd.read_csv(R + '/results.csv')
dev = df.groupby('algorithm').device.agg(lambda s: s.mode()[0])
DEV = {'cpu x8': 'CPU (8 jeder)', 'cuda: NVIDIA H100 PCIe': 'GPU (H100)'}
ov['total_h'] = (ov.train_time_total_s + ov.inference_time_total_s) / 3600
t = src([S + '/overall.csv', R + '/results.csv'])
t += r"""\begin{table}[htbp]
\centering
\caption{Časovna zahtevnost algoritmov. Mediani sta izračunani čez vseh 1080 učenj posameznega algoritma, skupni čas je vsota časa učenja in napovedovanja čez vsa učenja. Časi niso neposredno primerljivi med algoritmi, ki so tekli na različni strojni opremi.}
\label{tab:casi}
\footnotesize
\begin{tabular}{llrrr}
\hline
 & & \multicolumn{2}{c}{mediana na učenje [s]} & skupni čas \\
Algoritem & naprava & učenje & napovedovanje & [h] \\
\hline
"""
for _, r in ov.sort_values('total_h', ascending=False).iterrows():
    t += f"{IME[r.algorithm]} & {DEV[dev[r.algorithm]]} & {n(r.train_time_median_s,3)} & {n(r.inference_time_median_s,4)} & {n(r.total_h,1)} \\\\\n"
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
open(OUT + '/tab_casi.tex', 'w').write(t)

# 4. Po množicah (longtable)
pv = pd.read_csv(S + '/pivot.csv').set_index('dataset')
pv = pv.reindex(sorted(pv.index, key=str.lower))
t = src([S + '/pivot.csv'])
cols = ORDER
head = "Podatkovna množica & " + " & ".join(IME[c].replace('Naključni gozd', 'Nakl.\\ gozd') for c in cols) + " \\\\\n"
t += r"""{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{>{\raggedright\arraybackslash}p{3.3cm}""" + "r" * len(cols) + r"""}
\caption{Povprečni ROC-AUC po podatkovnih množicah (povprečje čez 15 učenj). Nakl.\ gozd označuje naključni gozd. Najboljša vrednost v vrstici je v krepkem tisku (primerjava na štiri decimalke), -- označuje neuspeh, zvezdica pa nasičeno podatkovno množico (najboljši ROC-AUC vsaj \num{0.995}).}
\label{tab:mnozice} \\
\hline
""" + head + "\\hline\n\\endfirsthead\n\\hline\n" + head + "\\hline\n\\endhead\n\\hline\n\\endfoot\n"
for ds, r in pv.iterrows():
    vals = r[cols].astype(float); best = np.round(vals.max(), 4)
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', r'\\allowbreak ', ds).replace('_', r'\_\allowbreak ') + (r'$^{*}$' if r.saturated else '')
    cells = []
    for c in cols:
        v = vals[c]
        if np.isnan(v): cells.append('--')
        else:
            s = n(v); cells.append(r'\textbf{' + s + '}' if np.round(v, 4) == best else s)
    t += name + " & " + " & ".join(cells) + " \\\\\n"
t += "\\end{longtable}\n}\n"
open(OUT + '/tab_mnozice.tex', 'w').write(t)

# 5. CD diagrami s slovenskimi oznakami
# src.stats.cd_diagram riše 7,5 palca široko sliko, ki se v nalogi skrči na
# širino besedila in pisava postane premajhna; za nalogo jo narišemo ožjo.
import matplotlib.pyplot as plt
_subplots = plt.subplots
plt.subplots = lambda *a, figsize=None, **k: _subplots(*a, figsize=(4.8, figsize[1] * 0.85), **k)
for f, suf in (('nemenyi.csv', ''), ('nemenyi_nonsaturated.csv', '_nenasicene')):
    o = ov if suf == '' else ovn
    fr = pd.read_csv(S + '/' + ('friedman.csv' if suf == '' else 'friedman_nonsaturated.csv')).iloc[0]
    mr = pd.Series(o.mean_rank.values, index=[IME[a] for a in o.algorithm])
    cd_diagram(mr, fr.CD, os.path.join(SLIKE, f'cc18_cd{suf}.png'))
    os.remove(os.path.join(SLIKE, f'cc18_cd{suf}.png'))

# 6. Dejstva za besedilo
print('friedman', pd.read_csv(S+'/friedman.csv').to_dict('records'))
print('friedman_ns', pd.read_csv(S+'/friedman_nonsaturated.csv').to_dict('records'))
rk = pd.read_csv(S + '/ranks.csv').set_index('dataset')
win = rk.eq(rk.min(axis=1), axis=0)
TREES = ['catboost','lightgbm','xgboost','random_forest']
for ds in rk.index[win[TREES].any(axis=1)]:
    print('drevo 1. mesto:', ds, [c for c in rk.columns if win.loc[ds, c]], pv.loc[ds].to_dict())
fm = pv[['tabicl','tabpfn']].max(axis=1); tr = pv[TREES].max(axis=1)
adv = (fm - tr).dropna().sort_values()
print('FM prednost top:', adv.tail(5).round(4).to_dict()); print('FM prednost dno:', adv.head(5).round(4).to_dict())
print('FM > drevesa na', (adv > 0).sum(), 'od', len(adv))
dd = (pv.tabicl - pv.tabpfn).dropna()
print('tabicl-tabpfn |d| mediana', round(dd.abs().median(),5), 'ekstremi', dd.sort_values().iloc[[0,1,-2,-1]].round(4).to_dict())
print('skupaj h', round(ov.total_h.sum(),1), ov.set_index('algorithm').total_h.round(2).to_dict())
e_ = df[df.error.notna()]; print(e_.groupby(['dataset','algorithm']).size().to_dict())
print('nasicene', int(pv.saturated.sum()))
