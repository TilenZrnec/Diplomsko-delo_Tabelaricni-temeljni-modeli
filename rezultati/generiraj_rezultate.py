"""Tabele, slike in številke za poglavje Rezultati.

Viri (repozitorij s kodo, results/runs/; kaj je kateri zagon: CLAUDE.md, Run registry):
  cc18_v4/combined         rezultat diplome na OpenML-CC18: naključni gozd in XGBoost iz
                           cc18_v4 (popravljena predobdelava, commit 8b601d1), LightGBM,
                           CatBoost, TabPFN-3.5 in TabICL iz cc18_v3 (commit 69cc51e)
  cc18_v3 / cc18_v2        primerjava TabPFN-3 in TabPFN-3.5 (samo številke v besedilu;
                           TabPFN-3.5 in TabICL v končnih rezultatih sta iz cc18_v3;
                           cc18_v2 iz gita, commit V2_COMMIT)
  medic3_raw_v2/combined,  Medic3 z vsemi 220 razredi oziroma s 160 najpogostejšimi
  medic3_160_v2/combined   (naključni gozd in XGBoost iz *_v2, ostali iz medic3_raw/_160)
  medic3_raw/diagnostics,  diagnostika LightGBM na Medic3 (fold 0)
  medic3_160/diagnostics

Zagon (iz korena repozitorija s kodo, v okolju tabular3.5):
    cd ../Diplomsko-delo_Koda
    python ../Diplomsko-delo_Tabelaricni-temeljni-modeli/rezultati/generiraj_rezultate.py \
        ../Diplomsko-delo_Tabelaricni-temeljni-modeli/rezultati \
        ../Diplomsko-delo_Tabelaricni-temeljni-modeli/slike
Na koncu izpiše številke, ki so povzete v besedilu poglavja.
"""
import io, json, os, re, shutil, subprocess, sys
# Stalen datum v metapodatkih PDF (matplotlib ga bere iz SOURCE_DATE_EPOCH; 2026-10-28),
# da ponoven zagon ne spremeni slik, ki se po vsebini niso spremenile.
os.environ.setdefault('SOURCE_DATE_EPOCH', '1793145600')
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from src.stats import cd_diagram

RUNS = 'results/runs'
R = RUNS + '/cc18_v4/combined'; S = R + '/summary'
V3 = RUNS + '/cc18_v3'; V4 = RUNS + '/cc18_v4'
V2 = 'results/runs/cc18_v2'; V2_COMMIT = '5a14045'
M_RAW = RUNS + '/medic3_raw_v2/combined'; M_160 = RUNS + '/medic3_160_v2/combined'
OUT = sys.argv[1]; SLIKE = sys.argv[2]
IME = {'random_forest': 'Naključni gozd', 'xgboost': 'XGBoost', 'lightgbm': 'LightGBM',
       'catboost': 'CatBoost', 'tabpfn': 'TabPFN-3.5', 'tabicl': 'TabICL'}
TREES = ['catboost', 'lightgbm', 'xgboost', 'random_forest']

sources = json.load(open(R + '/sources.json'))
C_NEW, C_BASE = sources['new']['git_commit'], sources['base']['git_commit']
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
def e(p): return f"\\num{{{p:.1e}}}" if p < 1e-3 else f"\\num{{{p:.4f}}}"
def nd(x): return "0" if x == 0 else (f"\\num{{{x:.1e}}}" if abs(x) < 1e-3 else n(x))
def src(files, what=None):
    what = what or (f"zbirka cc18_v4/combined: naključni gozd in XGBoost iz cc18_v4, commit {C_NEW}, "
                    f"ostali iz cc18_v3, commit {C_BASE}")
    return "% Samodejno iz Diplomsko-delo_Koda/" + ", ".join(files) + f" ({what}). Ne urejaj na roko.\n"
def write(name, text):
    open(os.path.join(OUT, name), 'w').write(text)


# 1. Skupna primerjava ---------------------------------------------------------
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
    t += (f"{IME[r.algorithm]} & {n(r.mean_roc_auc_complete)} & {n(r.mean_rank,2)} & {int(r.n_wins)} & "
          f"{int(r.n_failed_datasets)} & {n(q.mean_roc_auc_complete)} & {n(q.mean_rank,2)} & {int(q.n_wins)} \\\\\n")
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
write('tab_skupno.tex', t)

# 2. Wilcoxon + Holm -------------------------------------------------------------
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
\setlength{\tabcolsep}{2.5pt}
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
write('tab_wilcoxon.tex', t)

# 3. Časi ------------------------------------------------------------------------
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
write('tab_casi.tex', t)

# 4. Po množicah (longtable) -------------------------------------------------------
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
write('tab_mnozice.tex', t)

# 5. Različice knjižnic: oba zagona CC18 sta tekla v istem okolju (enak pip freeze), zato
#    tabelo, ki jo iz manifest.json ustvari scripts/gen_version_table.py, vzamemo iz cc18_v3.
assert json.load(open(V3 + '/manifest.json'))['pip_freeze'] == json.load(open(V4 + '/manifest.json'))['pip_freeze']
shutil.copyfile(V3 + '/summary/razlicice.tex', os.path.join(OUT, 'tab_razlicice.tex'))

# 7. Primerjava TabPFN-3 / TabPFN-3.5: cc18_v3 proti cc18_v2 (ista koda, le drug TabPFN) -----
a3 = pd.read_csv(V3 + '/results.csv')
ov3 = pd.read_csv(V3 + '/summary/overall.csv').set_index('algorithm')
a = v2('results.csv'); ov2 = v2('summary/overall.csv').set_index('algorithm')
kk = ['dataset', 'algorithm', 'fold']
mm = a3.merge(a[kk + ['roc_auc']], on=kk, suffixes=('', '_v2'))
mm['d'] = mm.roc_auc - mm.roc_auc_v2
both = mm.dropna(subset=['roc_auc', 'roc_auc_v2'])
# (tabele ni več; besedilo navaja le številke, ki jih skript izpiše spodaj)

# 8. Medic3 -----------------------------------------------------------------------
def medic(path):
    pdt = pd.read_csv(path + '/summary/per_dataset.csv').set_index('algorithm')
    res = pd.read_csv(path + '/results.csv')
    tt = pd.read_csv(path + '/summary/corrected_ttest_holm.csv')
    return pdt, res, tt
raw_t, raw_r, raw_tt = medic(M_RAW); m160_t, m160_r, m160_tt = medic(M_160)
MORDER = list(m160_t.sort_values('roc_auc_mean', ascending=False).index)
info = {}
for lab, res in (('raw', raw_r), ('160', m160_r)):
    r0 = res.iloc[0]
    info[lab] = dict(rows=int(r0.n_train + r0.n_test), n_train=int(r0.n_train), n_test=int(r0.n_test),
                     features=int(r0.n_features), classes=int(r0.n_classes), fits=len(res),
                     errors={a: str(x)[:120] for a, x in res[res.error.notna()].groupby('algorithm').error.first().items()})
def cell(tab, algo):
    r = tab.loc[algo]
    if r.n_failed == r.n_folds: return '\\multicolumn{1}{c}{--}'
    return f"{n(r.roc_auc_mean)} $\\pm$ {n(r.roc_auc_std)}"
def tcell(tab, algo, col, d):
    r = tab.loc[algo]
    return '--' if r.n_failed == r.n_folds or pd.isna(r[col]) else n(r[col], d)
t = src([M_RAW + '/summary/per_dataset.csv', M_160 + '/summary/per_dataset.csv'],
        what="naključni gozd in XGBoost iz medic3_raw_v2/medic3_160_v2, ostali iz medic3_raw/medic3_160")
t += r"""\begin{table}[htbp]
\centering
\caption{Rezultati na podatkovni množici Medic3 z vsemi """ + str(info['raw']['classes']) + r""" razredi in s """ + str(info['160']['classes']) + r""" najpogostejšimi razredi: povprečni ROC-AUC $\pm$ standardni odklon čez 15 učenj ter mediani časa učenja in napovedovanja na učenje. -- označuje neuspeh vseh učenj (TabPFN-3.5 sprejme največ 160 razredov). Časi primerjajo kombinacijo algoritma in strojne opreme (razdelek~\ref{sec:rez-casi}).}
\label{tab:medic3}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrr}
\hline
 & \multicolumn{3}{c}{""" + str(info['raw']['classes']) + r""" razredov} & \multicolumn{3}{c}{""" + str(info['160']['classes']) + r""" razredov} \\
 & & \multicolumn{2}{c}{mediana [s]} & & \multicolumn{2}{c}{mediana [s]} \\
Algoritem & ROC-AUC & učenje & napoved. & ROC-AUC & učenje & napoved. \\
\hline
"""
for algo in MORDER:
    t += (f"{IME[algo]} & {cell(raw_t, algo)} & {tcell(raw_t, algo, 'train_time_median_s', 1)} & "
          f"{tcell(raw_t, algo, 'inference_time_median_s', 1)} & {cell(m160_t, algo)} & "
          f"{tcell(m160_t, algo, 'train_time_median_s', 1)} & {tcell(m160_t, algo, 'inference_time_median_s', 1)} \\\\\n")
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
write('tab_medic3.tex', t)

def tt_rows(tt):
    rows = {}
    for _, r in tt.iterrows():
        a, b, d, wa, wb = r.a, r.b, r.mean_diff, r.wins_a, r.wins_b
        if MORDER.index(a) > MORDER.index(b): a, b, d, wa, wb = b, a, -d, wb, wa
        rows[(a, b)] = (d, wa, wb, r.p_holm, r.significant)
    return rows
traw, t160 = tt_rows(raw_tt), tt_rows(m160_tt)
t = src([M_RAW + '/summary/corrected_ttest_holm.csv', M_160 + '/summary/corrected_ttest_holm.csv'],
        what="naključni gozd in XGBoost iz medic3_raw_v2/medic3_160_v2, ostali iz medic3_raw/medic3_160")
t += r"""\begin{table}[htbp]
\centering
\caption{Popravljeni t-test za ponovljeno prečno preverjanje s Holmovim popravkom na podatkovni množici Medic3. Razlika je povprečna razlika ROC-AUC (prvi minus drugi algoritem) čez 15 učenj, zmage pa število učenj, v katerih je boljši prvi oziroma drugi. Z zvezdico so označene razlike, značilne pri $\alpha = 0{,}05$; -- označuje par z algoritmom, ki ni uspel.}
\label{tab:medic3-test}
\footnotesize
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lrrlrrl}
\hline
 & \multicolumn{3}{c}{""" + str(info['raw']['classes']) + r""" razredov} & \multicolumn{3}{c}{""" + str(info['160']['classes']) + r""" razredov} \\
Par algoritmov & razlika & zmage & $p_\mathrm{Holm}$ & razlika & zmage & $p_\mathrm{Holm}$ \\
\hline
"""
for key in sorted(t160, key=lambda k: (MORDER.index(k[0]), MORDER.index(k[1]))):
    row = f"{IME[key[0]]} -- {IME[key[1]]}"
    for W in (traw, t160):
        if key in W:
            d, x, y, p, s = W[key]
            row += f" & {n(d)} & {int(x)}:{int(y)} & {e(p)}{'$^{*}$' if s else ''}"
        else:
            row += " & -- & -- & --"
    t += row + " \\\\\n"
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
write('tab_medic3_test.tex', t)

# 9. CD diagrami s slovenskimi oznakami ---------------------------------------------
# src.stats.cd_diagram riše 7,5 palca široko sliko, ki se v nalogi skrči na
# širino besedila in pisava postane premajhna; za nalogo jo narišemo ožjo.
import matplotlib.pyplot as plt
_subplots = plt.subplots
plt.subplots = lambda *a, figsize=None, **kw: _subplots(*a, figsize=(4.8, figsize[1] * 0.85), **kw)
for suf in ('', '_nenasicene'):
    o = ov if suf == '' else ovn
    fr = pd.read_csv(S + '/' + ('friedman.csv' if suf == '' else 'friedman_nonsaturated.csv')).iloc[0]
    mr = pd.Series(o.mean_rank.values, index=[IME[x] for x in o.algorithm])
    cd_diagram(mr, fr.CD, os.path.join(SLIKE, f'cc18_cd{suf}.png'))
    os.remove(os.path.join(SLIKE, f'cc18_cd{suf}.png'))

# 10. Dejstva za besedilo -------------------------------------------------------------
print('== CC18 (cc18_v4/combined) ==')
print('N', N, 'nasicene', N_SAT, 'nenasicene', N_NS, 'vsi uspeli', N_OK, N_OK_NS, 'ucenj/algoritem', N_FITS, 'vseh ucenj', len(df))
print('napake', int(df.error.notna().sum()), 'raw_error', int(df.raw_error.notna().sum()))
print('rangi', ov.set_index('algorithm').mean_rank.round(4).to_dict(), 'zmage', ov.set_index('algorithm').n_wins.to_dict())
print('AUC', ov.set_index('algorithm').mean_roc_auc_complete.round(4).to_dict())
print('rangi_ns', ovn.set_index('algorithm').mean_rank.round(4).to_dict(), 'AUC_ns', ovn.set_index('algorithm').mean_roc_auc_complete.round(4).to_dict())
print('friedman', pd.read_csv(S + '/friedman.csv').to_dict('records'))
print('friedman_ns', pd.read_csv(S + '/friedman_nonsaturated.csv').to_dict('records'))
for f in ('wilcoxon_holm.csv', 'wilcoxon_holm_nonsaturated.csv'):
    w = pd.read_csv(S + '/' + f)
    print(f, 'neznacilni:', [(r.a, r.b, round(r.p_holm, 4)) for r in w.itertuples() if not r.significant])
    print(f, 'tabicl/tabpfn:', [(r.a, r.b, round(r.mean_diff, 4), r.wins_a, r.wins_b, r.p_holm) for r in w.itertuples() if {r.a, r.b} == {'tabicl', 'tabpfn'}])
    print(f, 'catboost/lightgbm, lightgbm/xgboost:', [(r.a, r.b, r.p_holm) for r in w.itertuples() if {r.a, r.b} in ({'catboost', 'lightgbm'}, {'lightgbm', 'xgboost'})])
for suf, o in (('', ov), ('_ns', ovn)):
    mr = o.set_index('algorithm').mean_rank; cd = pd.read_csv(S + ('/friedman.csv' if suf == '' else '/friedman_nonsaturated.csv')).CD.iloc[0]
    print('nemenyi' + suf, 'CD', round(cd, 3), 'razlike sosednjih:', {f'{a}-{b}': round(mr[b] - mr[a], 3) for a, b in zip(mr.index[:-1], mr.index[1:])},
          'tabicl-catboost', round(mr['catboost'] - mr['tabicl'], 3), 'catboost-lightgbm', round(mr['lightgbm'] - mr['catboost'], 3), 'lightgbm-rf', round(mr['random_forest'] - mr['lightgbm'], 3))
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
print('== TabPFN-3 / 3.5 (cc18_v3 proti cc18_v2) ==')
pd2 = a[a.error.isna()].groupby(['dataset', 'algorithm']).roc_auc.mean().unstack()
pv3 = pd.read_csv(V3 + '/summary/pivot.csv').set_index('dataset')
up = (pv3.tabpfn - pd2.tabpfn).dropna()
print('TabPFN-3.5 > TabPFN-3 na', int((up > 0).sum()), 'od', len(up), '| mediana', round(up.median(), 4), 'ekstremi', up.sort_values().iloc[[0, 1, -2, -1]].round(4).to_dict())
for algo in ('random_forest', 'tabicl', 'xgboost'):
    x = both[(both.algorithm == algo) & (both.d != 0)]
    print(algo, 'spremenjenih', len(x), 'max', round(x.d.abs().max(), 6) if len(x) else 0)
print('cc18_v3 rangi:', ov3.mean_rank.round(3).to_dict())
w2 = v2('summary/wilcoxon_holm.csv')
print('cc18_v2 tabicl-tabpfn:', w2[(w2.a.isin(['tabicl', 'tabpfn'])) & (w2.b.isin(['tabicl', 'tabpfn']))].to_dict('records'))
print('cc18_v2 rangi:', ov2.mean_rank.round(3).to_dict(), 'zmage', ov2.n_wins.to_dict())
print('== Medic3 ==')
print('info', info)
for lab, tab in (('raw', raw_t), ('160', m160_t)):
    print(lab, tab[['roc_auc_mean', 'roc_auc_std', 'n_failed', 'train_time_median_s', 'inference_time_median_s']].round(4).to_dict('index'))
for lab, tt in (('raw', raw_tt), ('160', m160_tt)):
    print(lab, 'parov', len(tt), 'znacilnih', int(tt.significant.sum()),
          'najsibkejsi', tt.sort_values('p_holm').iloc[-1][['a', 'b', 'mean_diff', 'wins_a', 'wins_b', 't', 'p_holm']].to_dict())
for lab, path in (('raw', RUNS + '/medic3_raw'), ('160', RUNS + '/medic3_160')):
    ind = pd.read_csv(path + '/diagnostics/lightgbm_4.7.0_independent.csv').set_index('setting')
    old = pd.read_csv(path + '/diagnostics/lightgbm_4.6.0_independent.csv').set_index('setting')
    print('LightGBM diag', lab, ind[['roc_auc', 'accuracy', 'majority_share_test', 'mean_max_proba', 'train_logloss_first',
                                     'train_logloss_last', 'max_abs_leaf_first', 'max_abs_leaf_max', 'iterations_in_model']].round(4).to_dict('index'),
          '| 4.6.0 default', old.loc['default', ['roc_auc', 'train_logloss_last']].round(4).to_dict())
    rr = pd.read_csv(path + '/results.csv'); lg = rr[rr.algorithm == 'lightgbm'].roc_auc
    print('   LightGBM vseh 15 ucenj: min', round(lg.min(), 4), 'max', round(lg.max(), 4))

# 11. Dodatni metriki: točnost in logaritemska izguba -----------------------------------
# metrics.csv naredi scripts/compute_metrics.py iz shranjenih napovedi (brez ponovnega
# učenja), povzetke in teste pa src.summary / src.stats z --metric v summary/<metrika>/.
MET = ['accuracy', 'log_loss']
MLAB = {'roc_auc': 'ROC-AUC', 'accuracy': 'točnost', 'log_loss': 'log. izguba'}
def msum(metric, f): return pd.read_csv(f"{S}/{metric}/{f}")
info_m = json.load(open(R + '/metrics_info.json'))
mov = {m: msum(m, 'overall.csv').set_index('algorithm') for m in MET}
movn = {m: msum(m, 'overall_nonsaturated.csv').set_index('algorithm') for m in MET}
assert all(mov[m][f'mean_{m}_complete'].notna().all() for m in MET), 'povprečja le na vseh množicah'
t = src([S + f'/{m}/{f}' for m in MET for f in ('overall.csv', 'overall_nonsaturated.csv')]
        + [R + '/metrics.csv'])
t += r"""\begin{table}[htbp]
\centering
\caption{Dodatni metriki na zbirki OpenML-CC18: povprečna točnost in povprečna logaritemska izguba čez podatkovne množice ter povprečni rang algoritma po posamezni metriki. Pri logaritemski izgubi je boljša nižja vrednost in rang 1 dobi najnižja izguba. Nasičenost množic je določena po ROC-AUC. Algoritmi so urejeni po povprečnem rangu po ROC-AUC (tabela~\ref{tab:skupno}).}
\label{tab:metrike}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrrrrrr}
\hline
 & \multicolumn{4}{c}{vseh """ + str(N) + r""" množic} & \multicolumn{4}{c}{""" + str(N_NS) + r""" nenasičenih množic} \\
 & \multicolumn{2}{c}{točnost} & \multicolumn{2}{c}{log. izguba} & \multicolumn{2}{c}{točnost} & \multicolumn{2}{c}{log. izguba} \\
Algoritem & povpr. & rang & povpr. & rang & povpr. & rang & povpr. & rang \\
\hline
"""
for algo in ORDER:
    cells = []
    for tab in (mov, movn):
        for m in MET:
            cells += [n(tab[m].loc[algo, f'mean_{m}_complete']), n(tab[m].loc[algo, 'mean_rank'], 2)]
    t += IME[algo] + ' & ' + ' & '.join(cells) + ' \\\\\n'
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
write('tab_metrike.tex', t)

def mmed(path, metric):
    return pd.read_csv(f"{path}/summary/{metric}/per_dataset.csv").set_index('algorithm')
mraw = {m: mmed(M_RAW, m) for m in MET}; m160 = {m: mmed(M_160, m) for m in MET}
def mcell(tab, algo, metric):
    r = tab.loc[algo]
    if r.n_failed == r.n_folds: return '\\multicolumn{1}{c}{--}'
    return f"{n(r[f'{metric}_mean'])} $\\pm$ {n(r[f'{metric}_std'])}"
t = src([M_RAW + f'/summary/{m}/per_dataset.csv' for m in MET] + [M_160 + f'/summary/{m}/per_dataset.csv' for m in MET],
        what="naključni gozd in XGBoost iz medic3_raw_v2/medic3_160_v2, ostali iz medic3_raw/medic3_160; "
             "metrike iz shranjenih napovedi, scripts/compute_metrics.py")
t += r"""\begin{table}[htbp]
\centering
\caption{Dodatni metriki na podatkovni množici Medic3: povprečna točnost in povprečna logaritemska izguba $\pm$ standardni odklon čez 15 učenj. Pri logaritemski izgubi je boljša nižja vrednost. -- označuje neuspeh vseh učenj. Algoritmi so urejeni kot v tabeli~\ref{tab:medic3}.}
\label{tab:medic3-metrike}
\footnotesize
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\hline
 & \multicolumn{2}{c}{""" + str(info['raw']['classes']) + r""" razredov} & \multicolumn{2}{c}{""" + str(info['160']['classes']) + r""" razredov} \\
Algoritem & točnost & log. izguba & točnost & log. izguba \\
\hline
"""
for algo in MORDER:
    t += (f"{IME[algo]} & {mcell(mraw['accuracy'], algo, 'accuracy')} & {mcell(mraw['log_loss'], algo, 'log_loss')} & "
          f"{mcell(m160['accuracy'], algo, 'accuracy')} & {mcell(m160['log_loss'], algo, 'log_loss')} \\\\\n")
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
write('tab_medic3_metrike.tex', t)

print('== Dodatni metriki ==')
print('metrics_info CC18', {k: info_m[k] for k in ('fits', 'fits_scored', 'max_abs_roc_auc_diff', 'max_row_sum_error')},
      info_m['computed_by'])
for path in (M_RAW, M_160):
    mi = json.load(open(path + '/metrics_info.json'))
    print('metrics_info', path, {k: mi[k] for k in ('fits', 'fits_scored', 'max_abs_roc_auc_diff')})
for m in MET:
    for lab, tab, suf in (('72', mov, ''), ('ns', movn, '_nonsaturated')):
        o = tab[m].sort_values('mean_rank')
        print(m, lab, 'rangi', o.mean_rank.round(4).to_dict(), 'povpr', o[f'mean_{m}_complete'].round(4).to_dict(),
              'zmage', o.n_wins.to_dict())
        fr = msum(m, f'friedman{suf}.csv').iloc[0]
        print('  friedman', {k: fr[k] for k in ('chi2_F', 'F_F', 'df1', 'df2', 'p_F', 'CD', 'reject_H0')})
        w = msum(m, f'wilcoxon_holm{suf}.csv')
        print('  neznacilni:', [(r.a, r.b, round(r.p_holm, 4)) for r in w.itertuples() if not r.significant])
        print('  tabicl/tabpfn:', [(r.a, r.b, round(r.mean_diff, 4), r.wins_a, r.wins_b, r.p_holm) for r in w.itertuples() if {r.a, r.b} == {'tabicl', 'tabpfn'}])
        fmt = w[(w.a.isin(['tabpfn', 'tabicl']) ^ w.b.isin(['tabpfn', 'tabicl']))]
        print('  FM proti drevesom vsi znacilni:', bool(fmt.significant.all()), 'najvecji p_holm', fmt.p_holm.max())
        mr = tab[m].mean_rank
        print('  nemenyi: tabicl-catboost', round(mr['catboost'] - mr['tabicl'], 3), 'CD', round(fr.CD, 3))
for lab, path, tabs in (('raw', M_RAW, mraw), ('160', M_160, m160)):
    for m in MET:
        print('Medic3', lab, m, tabs[m][[f'{m}_mean', f'{m}_std', 'n_failed']].round(4).to_dict('index'))
        tt = pd.read_csv(f"{path}/summary/{m}/corrected_ttest_holm.csv")
        print('   neznacilni:', [(r.a, r.b, round(r.mean_diff, 4), r.wins_a, r.wins_b, round(r.p_holm, 4)) for r in tt.itertuples() if not r.significant])
        print('   tabpfn/catboost:', [(r.a, r.b, round(r.mean_diff, 4), r.wins_a, r.wins_b, round(r.p_holm, 4)) for r in tt.itertuples() if {r.a, r.b} == {'tabpfn', 'catboost'}])

# 12. Materiali: lastnosti podatkovnih množic ----------------------------------------------
# results/datasets/<nabor>.csv naredi scripts/describe_datasets.py iz podatkov, kot jih
# naloži benchmark (koda 428167b, izhodi 92b1941).
DS = 'results/datasets'
props = pd.read_csv(DS + '/cc18.csv', dtype={'dataset_id': str})
props['categorical_share'] = props.n_categorical / props.n_features
assert len(props) == N and set(props.dataset) == set(pv.index)
def num(x, d):
    return f"\\num{{{int(round(x))}}}" if d == 0 and float(x).is_integer() else f"\\num{{{x:.{max(d, 1)}f}}}"
DESC = [('število primerov', 'n_rows', 0, 1), ('število atributov', 'n_features', 0, 1),
        ('število razredov', 'n_classes', 0, 1), ('delež kategoričnih atributov', 'categorical_share', 2, 1),
        ('delež manjkajočih vrednosti [\\%]', 'missing_share', 1, 100),
        ('razmerje najmanjšega in največjega razreda', 'imbalance_ratio', 3, 1)]
t = src([DS + '/cc18.csv'], what='scripts/describe_datasets.py, podatki, kot jih naloži benchmark')
t += r"""\begin{table}[htbp]
\centering
\caption{Lastnosti """ + str(N) + r""" podatkovnih množic zbirke OpenML-CC18: najmanjša vrednost, mediana in največja vrednost čez množice. Razmerje razredov je število primerov najmanjšega razreda, deljeno s številom primerov največjega. Lastnosti vseh množic so v dodatku~\ref{dod:lastnosti}.}
\label{tab:cc18-opis}
\small
\begin{tabular}{lrrr}
\hline
Lastnost & najmanj & mediana & največ \\
\hline
"""
for label, col, d, scale in DESC:
    s = props[col] * scale
    t += f"{label} & {num(s.min(), d)} & {num(s.median(), d)} & {num(s.max(), d)} \\\\\n"
t += "\\hline\n\\end{tabular}\n\\end{table}\n"
write('tab_cc18_opis.tex', t)

# Priloga: lastnosti vseh 72 množic.
t = src([DS + '/cc18.csv'], what='scripts/describe_datasets.py, podatki, kot jih naloži benchmark')
head = "Podatkovna množica & ID & primeri & atributi & kat. & razredi & manjk.\\ [\\%] \\\\\n"
t += r"""{\scriptsize
\setlength{\tabcolsep}{4pt}
\begin{longtable}{>{\raggedright\arraybackslash}p{4.6cm}rrrrrr}
\caption{Lastnosti podatkovnih množic zbirke OpenML-CC18: identifikator OpenML, število primerov, atributov, kategoričnih atributov in razredov ter delež manjkajočih vrednosti (celic) v odstotkih.}
\label{tab:cc18-lastnosti} \\
\hline
""" + head + "\\hline\n\\endfirsthead\n\\hline\n" + head + "\\hline\n\\endhead\n\\hline\n\\endfoot\n"
for _, r in props.sort_values('dataset', key=lambda s: s.str.lower()).iterrows():
    name = re.sub(r'(?<=[a-z])(?=[A-Z])', r'\\allowbreak ', r.dataset).replace('_', r'\_\allowbreak ')
    miss = '0' if r.missing_share == 0 else f"\\num{{{100 * r.missing_share:.2f}}}"
    t += (f"{name} & {r.dataset_id} & \\num{{{r.n_rows}}} & \\num{{{r.n_features}}} & {r.n_categorical} & "
          f"{r.n_classes} & {miss} \\\\\n")
t += "\\end{longtable}\n}\n"
write('tab_cc18_lastnosti.tex', t)

# Slika: velikost množic. Barvi sta reži 1 in 2 kategorične palete (modra, oranžna), ki
# sta preverjeni za razsevne diagrame; obliki znaka vrsto množice ločita tudi v sivinah.
C1, C2, INK, MUTED, GRID, AXIS = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e1e0d9', '#c3c2b7'
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED})
fig, ax = _subplots(figsize=(5.4, 3.3))
binary = props.n_classes == 2
for mask, col, mk, lab in ((binary, C1, 'o', 'binarne'), (~binary, C2, '^', 'z več razredi')):
    ax.scatter(props.n_rows[mask], props.n_features[mask], s=26, c=col, marker=mk, edgecolors='white',
               linewidths=0.6, label=f'{lab} ({int(mask.sum())})', zorder=3)
ax.set_xscale('log'); ax.set_yscale('log')
fmt = FuncFormatter(lambda v, _: f'{v:,.0f}'.replace(',', ' '))
ax.xaxis.set_major_formatter(fmt); ax.yaxis.set_major_formatter(fmt)
ax.axvline(10000, color=MUTED, lw=1.0, ls='--', zorder=1); ax.axhline(500, color=MUTED, lw=1.0, ls='--', zorder=1)
ax.text(10000 * 1.08, 2.6, 'TabPFN v2: do 10 000 primerov', color=MUTED, fontsize=7.5, va='bottom')
ax.text(420, 500 * 1.12, 'TabPFN v2: do 500 atributov', color=MUTED, fontsize=7.5, va='bottom')
for name, dx, dy, ha in (('CIFAR_10', -0.12, 0, 'right'), ('numerai28.6', 0, 0.16, 'center')):
    r = props.set_index('dataset').loc[name]
    ax.annotate(name, (r.n_rows, r.n_features), fontsize=7.5, color=INK,
                xytext=(r.n_rows * 10 ** dx, r.n_features * 10 ** dy), ha=ha, va='center')
ax.set_xlabel('število primerov'); ax.set_ylabel('število atributov')
ax.set_xlim(350, 2e5); ax.set_ylim(2, 6000)
ax.grid(True, which='major', color=GRID, lw=0.5, zorder=0)
for s in ('top', 'right'): ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc='upper left', fontsize=8, handletextpad=0.3)
fig.tight_layout()
fig.savefig(os.path.join(SLIKE, 'cc18_velikosti.pdf'))
plt.close(fig)

print('== Materiali (results/datasets) ==')
print('binarnih', int(binary.sum()), '| vsaj en kategoricni', int((props.n_categorical > 0).sum()),
      '| samo kategoricni', int((props.n_categorical == props.n_features).sum()),
      '| z manjkajocimi', int((props.missing_share > 0).sum()), 'najvec %', round(100 * props.missing_share.max(), 2),
      props.loc[props.missing_share.idxmax(), 'dataset'])
over = {'primerov > 10000': props.n_rows > 10000, 'atributov > 500': props.n_features > 500,
        'razredov > 10': props.n_classes > 10}
print({k: (int(v.sum()), sorted(props.dataset[v])) for k, v in over.items()})
print('zunaj obmocja TabPFN v2 (katerakoli meja):', int(np.logical_or.reduce(list(over.values())).sum()))
print('ekstremi', {c: (props.loc[props[c].idxmax(), 'dataset'], props[c].max()) for c in ('n_rows', 'n_features', 'n_classes')},
      'najmanj primerov', props.n_rows.min(), sorted(props.dataset[props.n_rows == props.n_rows.min()]))
print('razmerje razredov min', round(props.imbalance_ratio.min(), 4), props.loc[props.imbalance_ratio.idxmin(), 'dataset'])
for f in ('medic3', 'medic3_160'):
    print(f, pd.read_csv(f'{DS}/{f}.csv').iloc[0].to_dict())
