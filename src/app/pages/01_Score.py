import time
import requests
import streamlit as st

API_BASE = "https://mlet-5-tech-challenge.vercel.app/"

_SESSION = requests.Session()

def post_with_retry(url: str, payload: dict, *, retries: int = 3, timeout: int = 20, backoff: float = 1.7):
    """
    Faz POST com retries e backoff exponencial para lidar com:
    - cold start no Render
    - instabilidade momentânea
    - timeouts
    """
    last_err = None
    for attempt in range(retries):
        try:
            return _SESSION.post(url, json=payload, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as e:
            last_err = e
            time.sleep(backoff ** attempt)
    raise last_err

@st.cache_data(ttl=120, show_spinner=False)
def warmup():
    try:
        _SESSION.get(f"{API_BASE}/health", timeout=5)
    except Exception:
        pass


warmup()

st.title("🧠 Score - Risco de Defasagem", text_alignment='center')
st.caption("Preencha as variáveis e calcule o score (%) de risco de defasagem escolar do aluno.", text_alignment='center')

with st.form("score_form"):
    st.subheader("1 - Perfil do aluno", text_alignment="center")

    col1, col2 = st.columns(2)
    with col1:
        FASE = st.number_input("FASE", min_value=0, max_value=9, value=0, step=1)
        IDADE = st.number_input("IDADE", min_value=6, max_value=100, value=6, step=1)

    with col2:
        GENERO = st.selectbox(
            "GÊNERO",
            options=[0, 1],
            format_func=lambda x: "0 - Masculino" if x == 0 else "1 - Feminino",
        )
        ANO_INGRESSO = st.number_input("ANO DE INGRESSO", min_value=2000, max_value=2050, value=2020, step=1)

    col3 = st.columns(1)
    with col3[0]:
        INSTITUICAO_ENSINO = st.selectbox(
            "INSTITUIÇÃO DE ENSINO",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            format_func=lambda x: {
                1: "1 - Pública",
                2: "2 - Privada / Rede",
                3: "3 - Privada (Apadrinhamento)",
                4: "4 - Privada (Bolsa 100%)",
                5: "5 - Privada (Empresa parceira)",
                6: "6 - Concluiu 3º EM",
                7: "7 - Universitário formado(a)",
                8: "8 - Outros / Não informado",
            }[x],
        )

    st.subheader("2 - Indicadores e notas", text_alignment="center")

    st.text("Indicadores - Dimensão acadêmica:", text_alignment="center")
    col4, col5 = st.columns(2)
    with col4:
        IDA = st.number_input("IDA", min_value=0.00, max_value=10.00, value=0.0, step=0.01)
        IEG = st.number_input("IEG", min_value=0.00, max_value=10.00, value=0.0, step=0.01)

    with col5:
        NOTA_MATEM = st.number_input("NOTA MATEMÁTICA",  min_value=0.00, max_value=10.00,value=0.00, step=0.01)
        NOTA_PORT = st.number_input("NOTA PORTUGUÊS", min_value=0.00, max_value=10.00, value=0.00, step=0.01)

    st.text("Indicadores - Dimensão Psicossocial:")
    col6a, col6b = st.columns(2)
    with col6a:
        IAA = st.number_input("IAA", min_value=0.00, max_value=10.00, value=0.0, step=0.1)
    with col6b:
        IPS = st.number_input("IPS", min_value=0.00, max_value=10.00, value=0.0, step=0.01)

    st.text("Indicadores - Dimensão Psicopedagógica:")
    col7a, col7b = st.columns(2)
    with col7a:
        IPP = st.number_input("IPP", min_value=0.00, max_value=10.00, value=0.0, step=0.01)
    with col7b:
        IND_PV = st.number_input("IPV", min_value=0.00, max_value=10.00, value=0.0, step=0.01)

    st.text("Indicador GERAL:")
    col8 = st.columns(1)
    with col8[0]:
        INDE_ATUAL = st.number_input("ÚLTIMO INDE AVALIADO", min_value=0.00, max_value=10.00, value=0.0, step=0.01)

    submitted = st.form_submit_button("Calcular score")

if submitted:
    payload = {
        "FASE": FASE,
        "IDADE": IDADE,
        "GENERO": GENERO,
        "ANO_INGRESSO": ANO_INGRESSO,
        "INSTITUICAO_ENSINO": INSTITUICAO_ENSINO,
        "IAA": IAA,
        "IEG": IEG,
        "IPS": IPS,
        "IDA": IDA,
        "IND_PV": IND_PV,
        "IPP": IPP,
        "NOTA_MATEM": NOTA_MATEM,
        "NOTA_PORT": NOTA_PORT,
        "INDE_ATUAL": INDE_ATUAL,
    }

    try:
        with st.spinner("🔄 Consultando a API e calculando o score..."):
            r = post_with_retry(
                f"{API_BASE}/score",
                payload,
                retries=3,
                timeout=20,   
                backoff=1.7
            )

        if r.status_code != 200:
            st.error(f"Erro na API: {r.status_code} - {r.text}")
            st.stop()

        data = r.json()
        score = data.get("score_risco")
        pred = data.get("classe_predita")

        st.success("✅ Score calculado!")
        st.metric("Score de risco (%)", score)
        st.metric("Classe predita", pred)

        if score is not None:
            if score >= 70:
                st.warning("⚠️ Alto risco: priorizar acompanhamento.")
            elif score >= 40:
                st.info("ℹ️ Risco moderado: monitorar.")
            else:
                st.success("🟢 Baixo risco.")

        with st.expander("Ver payload enviado"):
            st.json(payload)

    except Exception as e:
        st.error("Não consegui obter resposta da API agora (lentidão/instabilidade).")
        st.caption(f"Detalhe técnico: {e}")
        st.stop()

# Rodapé
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