# Brasileirão Predictor AI 2026

O **Brasileirão Predictor AI** é uma plataforma de análise preditiva desenvolvida para o Campeonato Brasileiro de 2026. Este sistema utiliza Inteligência Artificial (Machine Learning) e Ciência de Dados para simular partidas, prever placares exatos e gerar tabelas de classificação dinâmicas com base em estatísticas históricas.

## 🚀 Como funciona

O projeto integra três pilares de dados:
1.  **Banco de Dados (PostgreSQL):** Armazena as informações estruturadas de times e metadados.
2.  **Arquivos CSV:** Dados de partidas reais e estatísticas detalhadas (posse de bola, passes, finalizações) extraídas de fontes como o FBref.
3.  **Pipeline de IA:** Utilizamos o algoritmo `RandomForestRegressor` do `scikit-learn`. O modelo é treinado com o histórico de 2025 para aprender padrões e realizar a regressão dos placares (gols mandante/visitante) dos jogos de 2026.

## 🛠️ Tecnologias Utilizadas
* **Python** (Pandas, Scikit-learn, SQLAlchemy)
* **Streamlit** (Interface Web)
* **PostgreSQL** (Banco de Dados)
* **CSS/HTML** (Estilização inspirada em Material Design)

## Referências das tabelas
https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol

https://fbref.com/en/comps/24/Serie-A-Stats

## 💻 Como Rodar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.10+ instalado.

1. **Clone o repositório ou navegue até a pasta do projeto.**

2. **Crie e ative um ambiente virtual:**
   ```bash
   python -m venv venv

   Rode a aplicação
   streamlit run src/app.py

   source venv/bin/activate  # No Windows: .\venv\Scripts\activate