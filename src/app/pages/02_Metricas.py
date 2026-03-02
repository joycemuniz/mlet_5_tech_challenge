from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from src.utils.config import METRICS_PATH, PREDICTIONS_PATH

st.title("📊 Métricas do Modelo", text_alignment='center')
st.caption("Leitura dos artefatos gerados em reports", text_alignment='center')

metrics_path = Path(METRICS_PATH)
pred_path = Path(PREDICTIONS_PATH)

if not metrics_path.exists():
    st.error(f"Não encontrei {metrics_path}. Rode a pipeline para gerar os reports.")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

f1 = float(metrics.get("f1", 0))
roc_auc = float(metrics.get("roc_auc", 0))
cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])

if not (isinstance(cm, list) and len(cm) == 2 and len(cm[0]) == 2 and len(cm[1]) == 2):
    st.error("confusion_matrix no metrics.json não está no formato 2x2.")
    st.write(cm)
    st.stop()

tn, fp = cm[0]
fn, tp = cm[1]
total = tn + fp + fn + tp

precision_1 = tp / (tp + fp) if (tp + fp) else 0.0
recall_1 = tp / (tp + fn) if (tp + fn) else 0.0
accuracy = (tp + tn) / total if total else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("F1-score", f"{f1:.3f}")
c2.metric("ROC-AUC", f"{roc_auc:.3f}")
c3.metric("Recall (classe 1)", f"{recall_1:.3f}")
c4.metric("Precision (classe 1)", f"{precision_1:.3f}")

st.subheader("Matriz de confusão (baseline do pipeline)")
st.dataframe(pd.DataFrame(cm, index=["Real 0", "Real 1"], columns=["Pred 0", "Pred 1"]))
st.caption(f"Total={total} | TN={tn}, FP={fp}, FN={fn}, TP={tp} | Accuracy={accuracy:.3f}")

if not pred_path.exists():
    st.warning(f"Não encontrei {pred_path}. Vou mostrar só o metrics.json.")
    st.stop()

df = pd.read_csv(pred_path)

required_cols = {"y_true", "y_pred", "score_risco"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"predictions.csv está sem colunas obrigatórias: {sorted(missing)}")
    st.write("Colunas encontradas:", df.columns.tolist())
    st.stop()

df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
df["y_pred"] = pd.to_numeric(df["y_pred"], errors="coerce").fillna(0).astype(int)
df["score_risco"] = pd.to_numeric(df["score_risco"], errors="coerce")

st.subheader("Distribuição do score de risco (%)")
scores = df["score_risco"].dropna()

if len(scores) > 0 and scores.max() <= 1.0:
    scores_plot = scores * 100
else:
    scores_plot = scores

fig = plt.figure()
plt.hist(scores_plot, bins=20)
plt.xlabel("Score (%)")
plt.ylabel("Frequência")
st.pyplot(fig)
plt.close(fig)

st.subheader("Distribuição das classes (real vs predita)")
c1, c2 = st.columns(2)
with c1:
    st.write("Real (y_true)")
    st.bar_chart(df["y_true"].value_counts().sort_index())
with c2:
    st.write("Predita (y_pred)")
    st.bar_chart(df["y_pred"].value_counts().sort_index())

st.subheader("Simulador de threshold (política de decisão)")
st.caption("Útil para discutir trade-off: reduzir FN (perder menos alunos em risco) vs reduzir FP (menos alarmes falsos).")

thr = st.slider("Threshold (%)", min_value=0, max_value=100, value=50, step=1)

df_sim = df.dropna(subset=["score_risco"]).copy()
score_for_thr = df_sim["score_risco"]
if score_for_thr.max() <= 1.0:
    score_for_thr = score_for_thr * 100

df_sim["y_pred_thr"] = (score_for_thr >= thr).astype(int)

tn2 = int(((df_sim["y_true"] == 0) & (df_sim["y_pred_thr"] == 0)).sum())
fp2 = int(((df_sim["y_true"] == 0) & (df_sim["y_pred_thr"] == 1)).sum())
fn2 = int(((df_sim["y_true"] == 1) & (df_sim["y_pred_thr"] == 0)).sum())
tp2 = int(((df_sim["y_true"] == 1) & (df_sim["y_pred_thr"] == 1)).sum())
total2 = tn2 + fp2 + fn2 + tp2

precision2 = tp2 / (tp2 + fp2) if (tp2 + fp2) else 0.0
recall2 = tp2 / (tp2 + fn2) if (tp2 + fn2) else 0.0
accuracy2 = (tp2 + tn2) / total2 if total2 else 0.0

d1, d2, d3 = st.columns(3)
d1.metric("Precision (classe 1)", f"{precision2:.3f}")
d2.metric("Recall (classe 1)", f"{recall2:.3f}")
d3.metric("Accuracy", f"{accuracy2:.3f}")

st.dataframe(
    pd.DataFrame([[tn2, fp2], [fn2, tp2]],
                 index=["Real 0", "Real 1"],
                 columns=["Pred 0", "Pred 1"])
)

st.subheader("Erros mais críticos (para análise)")
st.caption("FP: alto score mas era 0 | FN: baixo score mas era 1")

df_err = df_sim.copy()
df_err["score_pct"] = score_for_thr

fp_cases = df_err[(df_err["y_true"] == 0) & (df_err["y_pred_thr"] == 1)].sort_values("score_pct", ascending=False)
fn_cases = df_err[(df_err["y_true"] == 1) & (df_err["y_pred_thr"] == 0)].sort_values("score_pct", ascending=True)

tab1, tab2 = st.columns(2)
with tab1:
    st.write("🔴 Falsos Positivos (top 10 scores)")
    st.dataframe(fp_cases.head(10))
with tab2:
    st.write("🟠 Falsos Negativos (top 10 menores scores)")
    st.dataframe(fn_cases.head(10))

st.subheader("Amostra de previsões")
st.dataframe(df.head(20))

# =============================
# Rodapé e informações da desenvolvedora
# =============================

st.markdown(
    """
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        Desenvolvido por <b>Joyce Muniz</b><br>
        <a href='https://www.linkedin.com/in/joycemoliveira' target='_blank' style='text-decoration:none; color:gray;'>
            <img src='https://cdn-icons-png.flaticon.com/512/174/174857.png' width='18' style='vertical-align:middle; filter: grayscale(100%); margin-right:6px;'>
            joycemoliveira
        </a><br>
        <a href='https://github.com/joycemuniz' target='_blank' style='text-decoration:none; color:gray;'>
            <img src='https://icones.pro/wp-content/uploads/2021/06/icone-github-grise.png' width='23' style='vertical-align:middle; filter: grayscale(100%); margin-right:6px;'>
            joycemuniz
        </a>
    </div>
    """,
    unsafe_allow_html=True
)