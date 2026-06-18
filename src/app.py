import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.ensemble import RandomForestRegressor
import os
import numpy as np

# ==========================================
# 1. CONFIGURAÇÕES E ESTILO (Google Style)
# ==========================================
st.set_page_config(page_title="Brasileirão 2026", page_icon="⚽", layout="centered")

st.markdown("""
    <style>
        .google-card {
            background-color: #ffffff;
            border: 1px solid #dfe1e5;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            font-family: 'Roboto', Arial, sans-serif;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }
        .match-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 8px;
            margin-bottom: 8px;
        }
        .team-name {
            font-size: 16px;
            font-weight: 500;
            color: #202124;
            flex: 1;
        }
        .team-home { text-align: right; padding-right: 15px; }
        .team-away { text-align: left; padding-left: 15px; }
        .score-box {
            background-color: #f1f3f4;
            padding: 8px 16px;
            border-radius: 4px;
            font-size: 18px;
            font-weight: bold;
            color: #202124;
        }
        .match-status {
            font-size: 12px;
            color: #70757a;
            text-align: center;
            margin-bottom: 8px;
            text-transform: uppercase;
        }

        .google-table-container {
            font-family: 'Roboto', Arial, sans-serif;
            margin-top: 15px;
            border: 1px solid #dfe1e5;
            border-radius: 8px;
            overflow: hidden;
            background-color: #ffffff;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .google-table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
        }
        .google-table th {
            font-size: 12px;
            font-weight: 500;
            color: #70757a;
            padding: 12px 8px;
            border-bottom: 1px solid #dfe1e5;
            background-color: #f8f9fa;
        }
        .google-table td {
            font-size: 14px;
            color: #202124;
            padding: 10px 8px;
            border-bottom: 1px solid #e8eaed;
            vertical-align: middle;
        }
        .google-table tbody tr:hover {
            background-color: #f8f9fa;
        }
        .col-pos { width: 40px; text-align: center; color: #70757a; font-size: 13px; }
        .col-clube { text-align: left; font-weight: 500; }
        .col-num { width: 45px; text-align: center; }
        .col-pts { width: 50px; text-align: center; font-weight: bold; color: #000000; }
        
        .g4-row { border-left: 4px solid #1a73e8 !important; }
        .z4-row { border-left: 4px solid #d93025 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("Campeonato Brasileiro Série A")

# ==========================================
# 2. INTEGRAÇÃO E ENGENHARIA DE DADOS
# ==========================================
@st.cache_data
def carregar_dados():
    DATABASE_URI = "postgresql://postgres:12345678@localhost:5432/brasileirao_ai"
    try:
        engine = create_engine(DATABASE_URI)
        df_times = pd.read_sql("SELECT id_time, nome, sigla, estado, estadio_principal FROM times", engine)
    except SQLAlchemyError as e:
        st.error(f"Erro no PostgreSQL: {e}")
        st.stop()

    try:
        
        df_2025 = pd.read_csv('../data/partidas_2025.csv')
        df_2026 = pd.read_csv('../data/partidas_2026.csv')
        df_stats = pd.read_csv('../data/campeonato-brasileiro-estatisticas-full.csv')
    except Exception as e:
        st.error(f"Erro ao ler CSVs: {e}")
        st.stop()

    # ==========================================
    # CORREÇÃO DEFINITIVA: MAPEAMENTO DE NOMES
    # ==========================================
    # Copiamos exatamente os caracteres como estão no seu arquivo CSV
    mapeamento_nomes = {
        'Athletico–PR': 'Athletico-PR',   # Substituindo o traço longo
        'Atlético Mineiro': 'Atlético-MG',
        'RB Bragantino': 'Bragantino',
        'Botafogo–RJ': 'Botafogo',        # Substituindo o traço longo
        'Vasco da Gama': 'Vasco'
    }

    df_2025['Home'] = df_2025['Home'].replace(mapeamento_nomes)
    df_2025['Away'] = df_2025['Away'].replace(mapeamento_nomes)
    df_2026['Home'] = df_2026['Home'].replace(mapeamento_nomes)
    df_2026['Away'] = df_2026['Away'].replace(mapeamento_nomes)
    
    if 'clube' in df_stats.columns:
        df_stats['clube'] = df_stats['clube'].replace(mapeamento_nomes)

    # ==========================================
    # TRATAMENTO DE ESTATÍSTICAS E GOLS
    # ==========================================
    if df_stats['posse_de_bola'].dtype == object:
        df_stats['posse_de_bola'] = df_stats['posse_de_bola'].str.replace('%', '').astype(float)
    
    colunas_stats = ['chutes', 'chutes_no_alvo', 'posse_de_bola', 'passes']
    df_medias_times = df_stats.groupby('clube')[colunas_stats].mean().reset_index()
    media_geral = df_stats[colunas_stats].mean().to_dict()

    df_2025 = df_2025.dropna(subset=['Score']).copy()
    gols = df_2025['Score'].astype(str).str.extract(r'(\d+)\D+(\d+)')
    df_2025['gols_mandante'] = gols[0].astype(float)
    df_2025['gols_visitante'] = gols[1].astype(float)
    df_2025 = df_2025.dropna(subset=['gols_mandante', 'gols_visitante'])

    def integrar_dados(df_jogos, eh_treino=False):
        df = df_jogos.merge(df_times[['id_time', 'nome']], left_on='Home', right_on='nome', how='left')
        df.rename(columns={'id_time': 'id_mandante'}, inplace=True)
        df = df.merge(df_times[['id_time', 'nome']], left_on='Away', right_on='nome', how='left', suffixes=('', '_visitante'))
        df.rename(columns={'id_time': 'id_visitante'}, inplace=True)
        
        df['id_mandante'] = df['id_mandante'].fillna(0).astype(int)
        df['id_visitante'] = df['id_visitante'].fillna(0).astype(int)
        
        df = df.merge(df_medias_times, left_on='Home', right_on='clube', how='left')
        df.rename(columns={c: f"{c}_mandante" for c in colunas_stats}, inplace=True)
        df.drop('clube', axis=1, inplace=True)
        
        df = df.merge(df_medias_times, left_on='Away', right_on='clube', how='left')
        df.rename(columns={c: f"{c}_visitante" for c in colunas_stats}, inplace=True)
        df.drop('clube', axis=1, inplace=True)
        
        for c in colunas_stats:
            df[f"{c}_mandante"] = df[f"{c}_mandante"].fillna(media_geral[c])
            df[f"{c}_visitante"] = df[f"{c}_visitante"].fillna(media_geral[c])
            
        if eh_treino: df = df.dropna(subset=['gols_mandante', 'gols_visitante'])
        return df

    df_2025_full = integrar_dados(df_2025, eh_treino=True)
    df_2026_full = integrar_dados(df_2026, eh_treino=False)

    return df_times, df_2025_full, df_2026_full, colunas_stats

df_times, df_2025_full, df_2026_full, colunas_stats = carregar_dados()

# ==========================================
# 3. PIPELINE DE IA
# ==========================================
@st.cache_resource
def treinar_modelo(df_treino):
    features = ['id_mandante', 'id_visitante']
    for c in colunas_stats:
        features.extend([f"{c}_mandante", f"{c}_visitante"])
        
    X = df_treino[features]
    y = df_treino[['gols_mandante', 'gols_visitante']]

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    
    return modelo, features

modelo_rf, lista_features = treinar_modelo(df_2025_full)

# ==========================================
# 4. INTERFACE GOOGLE STYLE E LÓGICA
# ==========================================
rodadas_disponiveis = sorted(df_2026_full['Wk'].dropna().unique())
rodada_selecionada = st.selectbox("Rodada", rodadas_disponiveis, format_func=lambda x: f"Rodada {int(x)}")

df_rodada = df_2026_full[df_2026_full['Wk'] == rodada_selecionada].copy()
df_acumulado = df_2026_full[df_2026_full['Wk'] <= rodada_selecionada].copy()

aba_partidas, aba_classificacao = st.tabs(["Partidas", "Classificação (Simulada)"])

with aba_partidas:
    st.markdown("<br>", unsafe_allow_html=True)
    
    for index, jogo in df_rodada.iterrows():
        X_pred = jogo[lista_features].values.reshape(1, -1)
        previsao_gols = modelo_rf.predict(X_pred)[0]
        gols_mandante = max(0, int(np.round(previsao_gols[0])))
        gols_visitante = max(0, int(np.round(previsao_gols[1])))
        
        st.markdown(f"""
            <div class="google-card">
                <div class="match-status">Simulação IA • {jogo['Venue']}</div>
                <div class="match-row">
                    <div class="team-name team-home">{jogo['Home']}</div>
                    <div class="score-box">{gols_mandante} - {gols_visitante}</div>
                    <div class="team-name team-away">{jogo['Away']}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        with st.expander(f"Ver estatísticas de {jogo['Home']} vs {jogo['Away']}"):
            df_comparativo = pd.DataFrame({
                'Média Histórica': ['Chutes/Jogo', 'Chutes no Alvo', 'Posse de Bola', 'Passes'],
                jogo['Home']: [f"{jogo['chutes_mandante']:.1f}", f"{jogo['chutes_no_alvo_mandante']:.1f}", f"{jogo['posse_de_bola_mandante']:.1f}%", f"{jogo['passes_mandante']:.0f}"],
                jogo['Away']: [f"{jogo['chutes_visitante']:.1f}", f"{jogo['chutes_no_alvo_visitante']:.1f}", f"{jogo['posse_de_bola_visitante']:.1f}%", f"{jogo['passes_visitante']:.0f}"]
            })
            st.dataframe(df_comparativo, hide_index=True, use_container_width=True)

with aba_classificacao:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### Classificação Acumulada (Até a {int(rodada_selecionada)}ª Rodada)")
    
    estatisticas_times = {
        time: {'Pts': 0, 'J': 0, 'V': 0, 'E': 0, 'D': 0, 'GP': 0, 'GC': 0, 'SG': 0} 
        for time in df_times['nome'].tolist()
    }
    
    for index, row in df_acumulado.iterrows():
        X_pred = row[lista_features].values.reshape(1, -1)
        previsao = modelo_rf.predict(X_pred)[0]
        gols_m = max(0, int(np.round(previsao[0])))
        gols_v = max(0, int(np.round(previsao[1])))
        
        mandante = row['Home']
        visitante = row['Away']
        
        if mandante in estatisticas_times and visitante in estatisticas_times:
            estatisticas_times[mandante]['J'] += 1
            estatisticas_times[visitante]['J'] += 1
            estatisticas_times[mandante]['GP'] += gols_m
            estatisticas_times[mandante]['GC'] += gols_v
            estatisticas_times[visitante]['GP'] += gols_v
            estatisticas_times[visitante]['GC'] += gols_m
            
            if gols_m > gols_v:
                estatisticas_times[mandante]['Pts'] += 3
                estatisticas_times[mandante]['V'] += 1
                estatisticas_times[visitante]['D'] += 1
            elif gols_m == gols_v:
                estatisticas_times[mandante]['Pts'] += 1
                estatisticas_times[visitante]['Pts'] += 1
                estatisticas_times[mandante]['E'] += 1
                estatisticas_times[visitante]['E'] += 1
            else:
                estatisticas_times[visitante]['Pts'] += 3
                estatisticas_times[visitante]['V'] += 1
                estatisticas_times[mandante]['D'] += 1

    for time in estatisticas_times:
        estatisticas_times[time]['SG'] = estatisticas_times[time]['GP'] - estatisticas_times[time]['GC']
        
    df_tabela = pd.DataFrame.from_dict(estatisticas_times, orient='index').reset_index()
    df_tabela.rename(columns={'index': 'Clube'}, inplace=True)
    df_tabela = df_tabela.sort_values(by=['Pts', 'V', 'SG', 'GP'], ascending=[False, False, False, False]).reset_index(drop=True)
    
    html_tabela = '<div class="google-table-container">'
    html_tabela += '<table class="google-table">'
    html_tabela += '<thead><tr>'
    html_tabela += '<th class="col-pos">Pos.</th>'
    html_tabela += '<th class="col-clube">Clube</th>'
    html_tabela += '<th class="col-pts">Pts</th>'
    html_tabela += '<th class="col-num">J</th>'
    html_tabela += '<th class="col-num">V</th>'
    html_tabela += '<th class="col-num">E</th>'
    html_tabela += '<th class="col-num">D</th>'
    html_tabela += '<th class="col-num">GP</th>'
    html_tabela += '<th class="col-num">GC</th>'
    html_tabela += '<th class="col-num">SG</th>'
    html_tabela += '</tr></thead><tbody>'
    
    for i, row in df_tabela.iterrows():
        posicao = i + 1
        classe_borda = ""
        if posicao <= 4:
            classe_borda = 'class="g4-row"'
        elif posicao >= 17:
            classe_borda = 'class="z4-row"'
            
        html_tabela += f'<tr {classe_borda}>'
        html_tabela += f'<td class="col-pos">{posicao}</td>'
        html_tabela += f'<td class="col-clube">{row["Clube"]}</td>'
        html_tabela += f'<td class="col-pts">{row["Pts"]}</td>'
        html_tabela += f'<td class="col-num">{row["J"]}</td>'
        html_tabela += f'<td class="col-num">{row["V"]}</td>'
        html_tabela += f'<td class="col-num">{row["E"]}</td>'
        html_tabela += f'<td class="col-num">{row["D"]}</td>'
        html_tabela += f'<td class="col-num">{row["GP"]}</td>'
        html_tabela += f'<td class="col-num">{row["GC"]}</td>'
        html_tabela += f'<td class="col-num">{row["SG"]}</td>'
        html_tabela += '</tr>'
        
    html_tabela += '</tbody></table></div>'
    
    st.markdown(html_tabela, unsafe_allow_html=True)