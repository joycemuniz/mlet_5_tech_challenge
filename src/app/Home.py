import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import streamlit as st

st.set_page_config(page_title="MLET 5 - App", layout="wide")
st.title("📌 MLET 5 - Sistema de Score de Defasagem", text_alignment='center')
st.write("""
Este app possui duas seções:
- **Scoring**: calcular score de risco para novos inputs (via API).
- **Métricas**: acompanhar performance do modelo no conjunto de teste.
""")

st.info("Use o menu lateral para navegar entre as páginas.")

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