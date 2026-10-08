"""Tabele in slike za poglavje Rezultati, iz datotek zagona cc18_v3.

Zagon (iz korena repozitorija s kodo, v okolju tabular3.5):
    cd ../Diplomsko-delo_Koda
    python ../Diplomsko-delo_Tabelaricni-temeljni-modeli/rezultati/generiraj_iz_cc18_v3.py \
        ../Diplomsko-delo_Tabelaricni-temeljni-modeli/rezultati \
        ../Diplomsko-delo_Tabelaricni-temeljni-modeli/slike
Primerjavo s prejšnjim zagonom cc18_v2 (TabPFN-3) bere iz zgodovine gita
(commit V2_COMMIT), ker mape results/runs/cc18_v2 v delovnem drevesu ni več.
Na koncu izpiše še številke, ki so povzete v besedilu poglavja.
"""
import io, re, sys, os, shutil, subprocess
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.stats import cd_diagram
R = 'results/runs/cc18_v3'; S = R + '/summary'; COMMIT = '69cc51e'
V2 = 'results/runs/cc18_v2'; V2_COMMIT = '5a14045'
OUT = sys.argv[1]; SLIKE = sys.argv[2]
IME = {'random_forest': 'Naključni gozd', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
       'catboost': 'CatBoost', 'tabpfn': 'TabPFN-3.5', 'tabicl': 'TabICL'}
TREES = ['catboost', 'lightgbm', 'xgboost', 'random_forest']
ov = pd.read_csv(S + '/overall.csv'); ovn = pd.read_csv(S + '/overall_nonsaturated.csv')
ORDER = list(ov.algorithm)  # po povprečnem rangu
pv = pd.read_csv(S + '/pivot.csv').set_index('dataset')
df = pd.read_csv(R + '/results.csv')
N = len(pv); N_SAT = int(pv.saturated.sum()); N_NS = N - N_SAT
N_OK = int(pv[ORDER].notna().all(axis=1).sum())
N_OK_NS = int(pv.loc[~pv.saturated, ORDER].notna().all(axis=1).sum())
N_FITS = int(df.groupby('algorithm').size().iloc[0])
def v2(path):
    return pd.read_csv(io.StringIO(subprocess.check_output(['git', 'show', f'{V2_COMMIT}:{V2}/{path}'], text=True)))
def n(x, d=4): return f"\\num{{{x:.{d}f}}}"
def e(p):
    return f"\\num{{{p:.1e}}}" if p < 1e-3 else f"\\num{{{p:.4f}}}"
def nd(x):  # razlika: točna ničla kot 0, majhne razlike v eksponentnem zapisu
    return "0" if x == 0 else (f"\\num{{{x:.1e}}}" if abs(x) < 1e-3 else n(x))
def src(files, extra=''):
    return ("% Samodejno iz Diplomsko-delo_Koda/" + ", ".join(files)
            + f" (zagon cc18_v3, commit {COMMIT}{extra}). Ne urejaj na roko.\n")

# 1. Skupna primerjava
m = ovn.set_index('algorithm')
if N_OK == N and N_OK_NS == N_NS:
    povp = (f"Vsi algoritmi so uspeli na vseh podatkovnih množicah, zato sta povprečni ROC-AUC "
            f"in povprečni rang izračunana na vseh {N} oziroma {N_NS} množicah.")
else:
    povp = (f"Povprečni ROC-AUC je izračunan le na podatkovnih množicah, na katerih so uspeli vsi "
            f"algoritmi ({N_OK} od {N} oziroma {N_OK_NS} od {N_NS}), povprečni rang pa na vseh; "
            f"neuspeh na podatkovni množici prejme najslabši rang.")
t = src([S + '/overall.csv', S + '/overall_nonsaturated.csv'])
t += r"""\begin{table}[htbp]
\centering
\caption{Skupna primerjava algoritmov na zbirki OpenML-CC18. """ + povp + r""" Zmage štejejo prvo mesto, vključno z izenačitvami.}
\label{tab:skupno}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrr}
\hline
 & \multicolumn{4}{c}{vseh """ + str(N) + r""" množic} & \multicolumn{3}{c}{""" + str(N_NS) + r""" nenasičenih množic} \\
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
 & \multicolumn{3}{c}{vseh """ + str(N) + r""" množic} & \multicolumn{3}{c}{""" + str(N_NS) + r""" nenasičenih množic} \\
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
dev = df.groupby('algorithm').device.agg(lambda s: 'CPU (8 jeder)' if s.mode()[0].startswith('cpu') else 'GPU (H100)')
gpus = sorted(d.replace('cuda: NVIDIA ', '') for d in df.device.unique() if d.startswith('cuda'))
ov['total_h'] = (ov.train_time_total_s + ov.inference_time_total_s) / 3600
t = src([S + '/overall.csv', R + '/results.csv'])
t += r"""\begin{table}[htbp]
\centering
\caption{Časovna zahtevnost algoritmov. Mediani sta izračunani čez vseh """ + str(N_FITS) + r""" učenj posameznega algoritma, skupni čas je vsota časa učenja in napovedovanja čez vsa učenja. Časi niso neposredno primerljivi med algoritmi, ki so tekli na različni strojni opremi; temeljna modela sta tekla na dveh izvedbah grafične procesne enote (""" + ' in '.join(gpus) + r"""), drevesni ansambli pa na procesorjih različnih vozlišč gruče.}
\label{tab:casi}
\footnotesize
\begin{tabular}{llrrr}
\hline
 & & \multicolumn{2}{c}{mediana na učenje [s]} & skupni čas \\
Algoritem & naprava & učenje & napovedovanje & [h] \\
\hline
"""
for _, r in ov.sort_values('total_h', ascending=False).iterrows():
    t += f"{IME[r.algorithm]} & {dev[r.algorithm]} & {n(r.train_time_median_s,3)} & {n(r.inference_time_median_s,4)} & {n(r.total_h,1)} \\\\\n"
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
open(OUT + '/tab_casi.tex', 'w').write(t)

# 4. Po množicah (longtable)
pv = pv.reindex(sorted(pv.index, key=str.lower))
cols = ORDER
neuspeh = ' -- označuje neuspeh,' if pv[cols].isna().any().any() else ''
t = src([S + '/pivot.csv'])
head = "Podatkovna množica & " + " & ".join(IME[c].replace('Naključni gozd', 'Nakl.\\ gozd') for c in cols) + " \\\\\n"
t += r"""{\scriptsize
\setlength{\tabcolsep}{3pt}
\begin{longtable}{>{\raggedright\arraybackslash}p{3.3cm}""" + "r" * len(cols) + r"""}
\caption{Povprečni ROC-AUC po podatkovnih množicah (povprečje čez 15 učenj). Nakl.\ gozd označuje naključni gozd. Najboljša vrednost v vrstici je v krepkem tisku (primerjava na štiri decimalke),""" + neuspeh + r""" zvezdica pa označuje nasičeno podatkovno množico (najboljši ROC-AUC vsaj \num{0.995}).}
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

# 5. Različice knjižnic (tabela iz manifest.json, ustvari jo scripts/gen_version_table.py)
shutil.copyfile(S + '/razlicice.tex', OUT + '/tab_razlicice.tex')

# 6. Primerjava z zagonom cc18_v2 (TabPFN-3 namesto TabPFN-3.5, sicer ista koda in protokol)
a = v2('results.csv'); ov2 = v2('summary/overall.csv').set_index('algorithm')
k = ['dataset', 'algorithm', 'fold']
mm = df.merge(a[k + ['roc_auc']], on=k, suffixes=('', '_v2'))
mm['d'] = mm.roc_auc - mm.roc_auc_v2
both = mm.dropna(subset=['roc_auc', 'roc_auc_v2'])
t = src([R + '/results.csv', S + '/overall.csv'], extra=f'; cc18_v2 iz gita, commit {V2_COMMIT}')
t += r"""\begin{table}[htbp]
\centering
\caption{Primerjava z zagonom s prejšnjo zamrznitvijo različic knjižnic, v katerem je bil namesto modela TabPFN-3.5 uporabljen TabPFN-3, preostali algoritmi, podatkovne množice, delitve in semena pa so bili enaki. Razlike ROC-AUC so izračunane na posameznih učenjih (novi minus prejšnji zagon), na katerih sta uspela oba zagona (""" + f"{len(both)} od {len(mm)}" + r"""). V prejšnjem zagonu modela TabPFN-3 in TabICL na množici CIFAR\_10 nista uspela, kar je v njegovih povprečnih rangih upoštevano z najslabšim rangom.}
\label{tab:zagona}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrr}
\hline
 & \multicolumn{2}{c}{povprečni rang} & \multicolumn{3}{c}{razlika ROC-AUC na učenje} \\
Algoritem & prejšnji & novi & povprečje & največja $|\cdot|$ & spremenjenih \\
\hline
"""
for algo in ORDER:
    x = both[both.algorithm == algo]
    label = 'TabPFN-3 $\\rightarrow$ TabPFN-3.5' if algo == 'tabpfn' else IME[algo]
    t += (f"{label} & {n(ov2.loc[algo, 'mean_rank'], 2)} & {n(ov.set_index('algorithm').loc[algo, 'mean_rank'], 2)} & "
          f"{nd(x.d.mean())} & {nd(x.d.abs().max())} & {int((x.d != 0).sum())}/{len(x)} \\\\\n")
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
open(OUT + '/tab_zagona.tex', 'w').write(t)

# 7. CD diagrami s slovenskimi oznakami
# src.stats.cd_diagram riše 7,5 palca široko sliko, ki se v nalogi skrči na
# širino besedila in pisava postane premajhna; za nalogo jo narišemo ožjo.
import matplotlib.pyplot as plt
_subplots = plt.subplots
plt.subplots = lambda *a, figsize=None, **k: _subplots(*a, figsize=(4.8, figsize[1] * 0.85), **k)
for suf in ('', '_nenasicene'):
    o = ov if suf == '' else ovn
    fr = pd.read_csv(S + '/' + ('friedman.csv' if suf == '' else 'friedman_nonsaturated.csv')).iloc[0]
    mr = pd.Series(o.mean_rank.values, index=[IME[x] for x in o.algorithm])
    cd_diagram(mr, fr.CD, os.path.join(SLIKE, f'cc18_cd{suf}.png'))
    os.remove(os.path.join(SLIKE, f'cc18_cd{suf}.png'))

# 8. Dejstva za besedilo
print('N', N, 'nasicene', N_SAT, 'nenasicene', N_NS, 'vsi uspeli', N_OK, N_OK_NS, 'ucenj/algoritem', N_FITS)
print('napake', int(df.error.notna().sum()), 'raw_error', int(df.raw_error.notna().sum()))
print('friedman', pd.read_csv(S + '/friedman.csv').round(4).to_dict('records'))
print('friedman_ns', pd.read_csv(S + '/friedman_nonsaturated.csv').round(4).to_dict('records'))
rk = pd.read_csv(S + '/ranks.csv').set_index('dataset')
win = rk.eq(rk.min(axis=1), axis=0)
for ds in rk.index[win[TREES].any(axis=1)]:
    print('drevo 1. mesto:', ds, [c for c in rk.columns if win.loc[ds, c]], pv.loc[ds, ORDER].round(4).to_dict())
fm = pv[['tabicl', 'tabpfn']].max(axis=1); tr = pv[TREES].max(axis=1)
adv = (fm - tr).dropna().sort_values()
print('FM prednost top:', adv.tail(5).round(4).to_dict()); print('FM prednost dno:', adv.head(5).round(4).to_dict())
print('FM > drevesa na', int((adv > 0).sum()), 'od', len(adv), '| enako na 4 decimalke:', int((adv.round(4) == 0).sum()))
dd = (pv.tabicl - pv.tabpfn).dropna()
print('tabicl-tabpfn |d| mediana', round(dd.abs().median(), 5), 'ekstremi', dd.sort_values().iloc[[0, 1, -2, -1]].round(4).to_dict())
print('skupaj h', round(ov.total_h.sum(), 1), ov.set_index('algorithm').total_h.round(2).to_dict())
print('mediane', ov.set_index('algorithm')[['train_time_median_s', 'inference_time_median_s']].round(4).to_dict())
print('CIFAR_10', pv.loc['CIFAR_10', ORDER].round(4).to_dict())
pd2 = a[a.error.isna()].groupby(['dataset', 'algorithm']).roc_auc.mean().unstack()
up = (pv.tabpfn - pd2.tabpfn).dropna()
print('TabPFN-3.5 > TabPFN-3 na', int((up > 0).sum()), 'od', len(up), '| mediana', round(up.median(), 4), 'ekstremi', up.sort_values().iloc[[0, 1, -2, -1]].round(4).to_dict())
for algo in ('random_forest', 'tabicl'):
    x = both[(both.algorithm == algo) & (both.d != 0)]
    print(algo, 'spremenjenih', len(x), 'max', round(x.d.abs().max(), 6), 'nabori', x.groupby('dataset').d.apply(lambda s: s.abs().max()).sort_values().tail(3).round(6).to_dict())
w2 = v2('summary/wilcoxon_holm.csv')
print('cc18_v2 tabicl-tabpfn:', w2[(w2.a.isin(['tabicl', 'tabpfn'])) & (w2.b.isin(['tabicl', 'tabpfn']))].to_dict('records'))
print('cc18_v2 rangi:', ov2.mean_rank.round(3).to_dict(), 'zmage', ov2.n_wins.to_dict())
