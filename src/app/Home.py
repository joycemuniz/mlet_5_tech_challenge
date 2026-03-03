import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import streamlit as st

st.set_page_config(page_title="MLET 5 - App", layout="wide")
st.title("📌 MLET 5 - Sistema de Score de Defasagem", text_alignment='center')

st.divider()

# ====================================
# Sobre a Associação
# ====================================

st.subheader("Sobre a Associação Passos Mágicos", text_alignment="center")

st.markdown("""
A **Associação Passos Mágicos** possui mais de **33 anos de atuação**, promovendo a transformação da vida de crianças e jovens de baixa renda e ampliando suas oportunidades educacionais e de desenvolvimento.

A organização atua no apoio ao **desenvolvimento educacional e social de crianças e jovens em situação de vulnerabilidade**, acompanhando sua trajetória escolar e oferecendo suporte por meio de **reforço educacional, acompanhamento pedagógico e apoio psicossocial**.

A iniciativa foi idealizada por **Michelle Flues e Dimetri Ivanoff** e teve início em **1992**, atuando inicialmente em orfanatos no município de **Embu-Guaçu (SP)**.
""")

# ====================================
# Botão centralizado
# ====================================

col1, col2, col3, = st.columns([1,1,1])

with col2:  
    st.link_button(
        "🔎 Saiba mais sobre a Associação Passos Mágicos",
        "https://passosmagicos.org.br/"
    )

st.divider()

# ====================================
# Objetivo da solução
# ====================================

st.subheader(" 🎯 Objetivo da Solução",text_alignment="center")

st.markdown("""
Este projeto tem como objetivo **desenvolver um score de risco de defasagem escolar (0–100%)**, utilizando variáveis relacionadas ao **perfil do aluno** e a **indicadores acadêmicos, psicossociais e psicopedagógicos**.

A solução busca **apoiar a priorização de acompanhamento pedagógico e o monitoramento preventivo**, contribuindo para que alunos com maior risco recebam atenção educacional de forma antecipada e direcionada.

A aplicação integra **Machine Learning, API de predição e interface interativa**, permitindo que usuários consultem rapidamente o risco estimado a partir das informações do aluno.
""")

st.divider()

st.info("Use o menu lateral para navegar entre as páginas.")
st.write("""
- **Scoring**: calcular score de risco para novos inputs (via API).
- **Métricas**: acompanhar performance do modelo no conjunto de teste.
""")
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