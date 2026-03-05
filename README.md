# 🎓 MLET 5 -- Modelo de Score de Risco de Defasagem Escolar

Machine Learning Engineering -- FIAP

------------------------------------------------------------------------

## 📌 Objetivo

Desenvolver um modelo de Machine Learning capaz de prever o risco de
defasagem escolar de alunos com base em indicadores acadêmicos,
psicossociais e psicopedagógicos.

O projeto contempla:

-   Pipeline de dados estruturada
-   Split temporal (treino 2022-2023 \| teste 2024)
-   Modelo Random Forest
-   API REST com FastAPI
-   Interface interativa com Streamlit
-   Estrutura modular escalável

------------------------------------------------------------------------

## 🧠 Problema de Negócio

Identificar alunos com maior risco de defasagem escolar permite:

-   Ações preventivas
-   Priorização de acompanhamento
-   Melhor alocação de recursos pedagógicos
-   Redução de evasão e impacto educacional

O modelo retorna:

-   Classe predita (0 ou 1)
-   Score de risco (%)

------------------------------------------------------------------------

## 🏗 Arquitetura do Projeto

mlet_5_tech_challenge/ 
│ 
├── .github/ 
│   └── workflows/ 
│       └── ci.yml 
│ 
├── .streamlit/ 
│   └── config.toml 
│ 
├── data/ 
│   ├── raw/ 
│   │   └── BASE DE DADOS PEDE 2024 - DATATHON.xlsx 
│   ├── interim/ 
│   │   └── dataset_concatenado.csv 
│   └── refined/ 
│       └── dados_modelo.csv 
│ 
├── models/ 
│   └── model.pkl 
│ 
├── notebooks/ 
│   ├── 01_eda.ipynb 
│   ├── 02_preprocessing.ipynb 
│   └── 03_modelagem.ipynb 
│ 
├── reports/ 
│   ├── metrics.json 
│   └── predictions.csv 
│ 
├── src/ 
│   ├── api/ 
│   │   └── app.py 
│   │ 
│   ├── app/ 
│   │   ├── Home.py 
│   │   └── pages/ 
│   │       ├── 01_Score.py 
│   │       └── 02_Metricas.py 
│   │ 
│   ├── modeling/ 
│   │   ├── train.py 
│   │   └── evaluate.py 
│   │ 
│   ├── pipeline/ 
│   │   ├── data_consolidation.py 
│   │   ├── preprocessing.py 
│   │   ├── feature_engineering.py 
│   │   └── run_pipeline.py 
│   │ 
│   └── utils/ 
│       ├── config.py 
│       ├── metrics.py 
│       └── utils.py 
│ 
├── test/ 
│   └── testes automatizados 
│ 
├── Dockerfile 
├── render.yaml 
├── requirements.txt 
├── requirements-api.txt 
├── runtime.txt 
└── README.md 



------------------------------------------------------------------------

## 🔄 Pipeline de Dados

### 1️⃣ Consolidação

-   Leitura de múltiplas sheets (2022, 2023, 2024)
-   Padronização de colunas
-   Criação da variável target

### 2️⃣ Pré-processamento

-   Encoding de variáveis categóricas
-   Tratamento de nulos
-   Criação de INDE_ATUAL
-   Imputação controlada de IPP (sem vazamento temporal)
-   Remoção de fases sem label (8 e 9)

### 3️⃣ Split Temporal

Treino: - 2022 - 2023

Teste: - 2024

Evita leakage e simula cenário real de produção.

------------------------------------------------------------------------

## 🤖 Modelo

Algoritmo utilizado: - RandomForestClassifier

Configuração: - n_estimators = 200 - class_weight = balanced -
random_state = 42

------------------------------------------------------------------------

## 📊 Métricas (Teste 2024)

  Métrica              Valor
  -------------------- --------
  F1-score             0.834
  ROC-AUC              0.887
  Recall Classe 1      0.914
  Precision Classe 1   0.767
  Acurracy             0.816


O modelo apresenta forte capacidade discriminativa e bom equilíbrio
entre precision e recall.

------------------------------------------------------------------------

## 🚀 Como Executar

### 1️⃣ Criar ambiente virtual

    python -m venv .venv

### 2️⃣ Ativar (Windows)

    .venv\Scripts\activate

### 3️⃣ Instalar dependências

    pip install -r requirements.txt

### 4️⃣ Rodar pipeline

    python -m src.pipeline.run_pipeline

Gera: - models/model.pkl - reports/metrics.json -
reports/predictions.csv

### 5️⃣ Rodar API

    python -m uvicorn src.api.app:app --reload

### 6️⃣ Rodar Frontend (Streamlit)

    streamlit run src/app/home.py

------------------------------------------------------------------------

## 🔌 Endpoint da API

### POST /score

Entrada:

``` json
{
  "FASE": 7,
  "IDADE": 16,
  "GENERO": 1,
  "ANO_INGRESSO": 2020,
  "INSTITUICAO_ENSINO": 2,
  "IAA": 7.0,
  "IEG": 8.0,
  "IPS": 7.2,
  "IDA": 6.8,
  "IND_PV": 0.12,
  "IPP": 7.0,
  "IPP_IMPUTADO": 0,
  "NOTA_MATEM": 6.5,
  "NOTA_PORT": 6.8,
  "INDE_ATUAL": 7.4
}
```

Resposta:

``` json
{
  "classe_predita": 1,
  "score_risco": 72.3
}
```

------------------------------------------------------------------------

## 🧪 Boas Práticas Aplicadas

✔ Split temporal realista\
✔ Sem vazamento de dados\
✔ Estrutura modular\
✔ Feature engineering controlado\
✔ API desacoplada\
✔ Interface separada\
✔ Métricas versionadas

------------------------------------------------------------------------

## 🌐 Deploy

- **API**: [Render](https://mlet-5-tech-challenge.onrender.com) 
- **App Streamlit**: [Streamlit Cloud](https://score-defasagem.streamlit.app/)

------------------------------------------------------------------------

## 🎥 Demonstração em Vídeo

📽️ Link: [Youtube]()

------------------------------------------------------------------------

## Próximas Evoluções (sugestões)

- Automação de retreino
- Versionamento de modelos
- Observabilidade e alertas para degradação de performance

------------------------------------------------------------------------

## 👩‍💻 Autora

Joyce Muniz de Oliveira\
Machine Learning Engineering -- FIAP\
🔗 LinkedIn: https://www.linkedin.com/in/joycemoliveira
