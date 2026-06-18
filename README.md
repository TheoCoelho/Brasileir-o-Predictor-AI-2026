# ⚽ Brasileirão Predictor AI 2026

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue)

Um sistema avançado de Ciência de Dados e Machine Learning desenvolvido para simular, prever e analisar os resultados das partidas do Campeonato Brasileiro Série A. O projeto utiliza estatísticas reais de elencos e jogadores para prever o desfecho do campeonato rodada a rodada.

---

## 🎯 Funcionalidades

* **🔮 Simulação de Partidas:** Previsão de placares e tendências de resultado (Vitória/Empate/Derrota) baseada no histórico tático das equipes.
* **📊 Análise Tática Pré-Jogo:** Comparativo direto entre os times exibindo métricas como Posse de Bola, Gols Pró, Desarmes Certos, Chutes Sofridos e Eficiência do Goleiro.
* **🏆 Classificação Simulada:** Geração automática e dinâmica da tabela do Brasileirão, acumulando pontos com base nas previsões da Inteligência Artificial.
* **📈 Validação de Acurácia:** Exibição em tempo real da assertividade do modelo, comparando as previsões com os jogos que já aconteceram na vida real.

---

## 🧠 Arquitetura do Modelo de Machine Learning

Este projeto não se baseia apenas em médias de gols, mas sim no **comportamento tático e na eficiência coletiva dos elencos**.

1.  **Modelo:** `GradientBoostingRegressor` com `MultiOutputRegressor` (para prever gols do mandante e visitante simultaneamente).
2.  **Engenharia de Features (Deltas):** A IA não avalia as equipes isoladamente. O modelo calcula a diferença matemática (Delta) entre as estatísticas do Mandante e do Visitante. Se o *Delta de Desarmes* é muito alto para o mandante, a IA entende a superioridade defensiva no confronto direto.
3.  **Lógica de Empates (Thresholding):** O futebol é um esporte de placares baixos e alta taxa de empates. O modelo utiliza um limiar de confiança (`0.45`). Se a diferença de gols previstos entre as equipes for menor que esse limiar, a IA classifica o jogo como empate, aumentando drasticamente o realismo da tabela.

### Dados Utilizados (Features)
* `Poss`: Posse de bola média.
* `Gls` / `Ast`: Força ofensiva (Gols e Assistências totais do elenco).
* `Save%`: Eficiência do goleiro (% de defesas).
* `SoTA`: Chutes no alvo sofridos.
* `TklW`: Desarmes certos.
* `CrdY`: Cartões amarelos.

---
## Referências das tabelas
https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol

https://fbref.com/en/comps/24/Serie-A-Stats

## Como Executar o Projeto Localmente
1. Pré-requisitos
- Python 3.10 ou superior.
- PostgreSQL instalado e rodando localmente (banco brasileirao_ai configurado).
- Git.

2. Clonando o Repositório
Bash
git clone [https://github.com/SEU_USUARIO/Brasileiro-Predictor-AI.git](https://github.com/SEU_USUARIO/Brasileiro-Predictor-AI.git)
cd Brasileiro-Predictor-AI
3. Configurando o Ambiente Virtual
Crie e ative um ambiente virtual para instalar as dependências isoladamente:

   **No Mac/Linux:**

   Bash

   python3 -m venv .venv

   source .venv/bin/activate

   **No Windows:**

      DOS

      python -m venv .venv

      .venv\Scripts\activate
4. Instalando as Dependências
   Bash

   pip install -r requirements.txt

   (Caso não tenha o arquivo requirements.txt, instale manualmente rodando: pip install streamlit pandas sqlalchemy psycopg2-binary scikit-learn numpy)

5. Executando a Aplicação
   Com o ambiente ativado, inicie o servidor do Streamlit:

   Bash

   streamlit run src/app.py

   A aplicação abrirá automaticamente no seu navegador no endereço http://localhost:8501.

## 🛠️ Tecnologias Utilizadas
Front-end & Dashboard: Streamlit

Manipulação de Dados: Pandas e NumPy

Machine Learning: Scikit-Learn

Banco de Dados: PostgreSQL via SQLAlchemy

## 📁 Estrutura do Projeto

```text
PROJETO_FUTEBOL/
├── data/                                 # Diretório de dados brutos
│   ├── partidas_2025.csv                 # Dados de treino (Temporada passada)
│   ├── partidas_2026.csv                 # Dados de teste/simulação (Temporada atual)
│   ├── estatisticas-padrao-do-elenco.csv # Dados táticos
│   ├── estatisticas-diversas-do-elenco.csv
│   └── goleiros-do-elenco.csv
├── src/                                  # Código-fonte
│   └── app.py                            # Aplicação principal Streamlit e pipeline de IA
├── .venv/                                # Ambiente virtual Python
├── .gitignore                            # Arquivos ignorados pelo Git
├── requirements.txt                      # Dependências do projeto
└── README.md                             # Documentação