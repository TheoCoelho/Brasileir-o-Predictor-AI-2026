import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import numpy as np

# ==========================================
# 1. CONFIGURAÇÕES E ESTILO PREMIUM SPORTS (Dark Mode)
# ==========================================
st.set_page_config(page_title="Brasileirão AI 2026", page_icon="⚽", layout="centered")

st.markdown("""
    <style>
        /* Customizações globais do Streamlit para casar com o Dark Mode */
        .stApp { background-color: #0b0e14; color: #f8fafc; }
        h1, h2, h3, p { font-family: 'Inter', sans-serif; color: #ffffff !important; }
        
        /* Estilização dos Cards de Jogos (Estilo Sofascore Live) */
        .sports-card {
            background: linear-gradient(135deg, #131722 0%, #1c2230 100%);
            border: 1px solid #2a3547;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 16px;
            font-family: 'Inter', sans-serif;
            box-shadow: 0 4px 15px rgba(0,0,0,0.25);
            transition: all 0.2s ease-in-out;
        }
        .sports-card:hover {
            border-color: #00ff87;
            transform: translateY(-2px);
        }
        .match-status {
            font-size: 11px; color: #94a3b8; font-weight: 600;
            text-align: center; margin-bottom: 12px;
            text-transform: uppercase; letter-spacing: 1.5px;
        }
        .match-row { display: flex; justify-content: space-between; align-items: center; }
        .team-name { font-size: 16px; font-weight: 600; color: #ffffff; flex: 1; }
        .team-home { text-align: right; padding-right: 20px; }
        .team-away { text-align: left; padding-left: 20px; }
        
        /* Box do Placar Analítico */
        .score-container { display: flex; flex-direction: column; align-items: center; min-width: 110px; }
        .score-box {
            background-color: #0b0e14; border: 2px solid #2a3547;
            padding: 8px 20px; border-radius: 8px;
            font-size: 22px; font-weight: 800; color: #00ff87;
            box-shadow: inset 0 0 10px rgba(0,255,135,0.05);
        }
        .real-score { font-size: 11px; color: #f97316; font-weight: 700; margin-top: 8px; letter-spacing: 0.5px; text-transform: uppercase; }

        /* Estilização da Tabela de Classificação Profissional */
        .sports-table-container {
            margin-top: 15px; border: 1px solid #2a3547;
            border-radius: 12px; overflow: hidden;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .sports-table { width: 100%; border-collapse: collapse; text-align: left; background-color: #131722; }
        .sports-table th {
            font-size: 11px; font-weight: 600; color: #94a3b8;
            padding: 14px 10px; border-bottom: 2px solid #2a3547;
            background-color: #1c2230; text-transform: uppercase; letter-spacing: 0.5px;
        }
        .sports-table td { font-size: 14px; color: #f8fafc; padding: 12px 10px; border-bottom: 1px solid #2a3547; vertical-align: middle; }
        .sports-table tbody tr:hover { background-color: #1c2230; }
        
        .col-pos { width: 40px; text-align: center; color: #94a3b8; font-size: 12px; font-weight: bold; }
        .col-clube { text-align: left; font-weight: 600; color: #ffffff; }
        .col-num { width: 45px; text-align: center; color: #cbd5e1; }
        .col-pts { width: 50px; text-align: center; font-weight: 700; color: #00ff87 !important; }
        
        /* Indicadores de G4 e Z4 no estilo Zona de Classificação */
        .g4-mark { border-left: 5px solid #10b981 !important; }
        .z4-mark { border-left: 5px solid #ef4444 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("🏆 Brasileirão Predictor AI")

# ==========================================
# 2. INTEGRAÇÃO E ENGENHARIA DE DADOS
# ==========================================
@st.cache_data
def carregar_dados():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.basename(base_dir) == 'src': base_dir = os.path.dirname(base_dir)
    data_dir = os.path.join(base_dir, 'data')

    try:
        df_2025 = pd.read_csv(os.path.join(data_dir, 'partidas_2025.csv'))
        df_2026 = pd.read_csv(os.path.join(data_dir, 'partidas_2026.csv'))
        df_elenco = pd.read_csv(os.path.join(data_dir, 'estatisticas-padrao-do-elenco.csv'), header=1)
        df_goleiros = pd.read_csv(os.path.join(data_dir, 'goleiros-do-elenco.csv'), header=1)
        df_diversas = pd.read_csv(os.path.join(data_dir, 'estatisticas-diversas-do-elenco.csv'), header=1)
    except Exception as e:
        st.error(f"Erro ao ler CSVs: {e}")
        st.stop()

    # Normalização Extrema de Nomes
    mapeamento = {
        'Athletico–PR': 'Athletico-PR', 'Athletico PR': 'Athletico-PR', 
        'Atlético Mineiro': 'Atlético-MG', 'Atletico MG': 'Atlético-MG',
        'RB Bragantino': 'Bragantino', 'Botafogo–RJ': 'Botafogo', 'Botafogo RJ': 'Botafogo',
        'Vasco da Gama': 'Vasco', 'Sao Paulo': 'São Paulo'
    }
    for df in [df_2025, df_2026]:
        df['Home'] = df['Home'].replace(mapeamento)
        df['Away'] = df['Away'].replace(mapeamento)
    for df in [df_elenco, df_goleiros, df_diversas]:
        df['Squad'] = df['Squad'].replace(mapeamento)

    # Conversão Numérica
    for col in ['Poss', 'Gls', 'Ast']: df_elenco[col] = pd.to_numeric(df_elenco[col], errors='coerce').fillna(0)
    for col in ['Save%', 'SoTA']: df_goleiros[col] = pd.to_numeric(df_goleiros[col], errors='coerce').fillna(50)
    for col in ['TklW', 'CrdY']: df_diversas[col] = pd.to_numeric(df_diversas[col], errors='coerce').fillna(0)

    # Merge tático
    df_taticas = df_elenco[['Squad', 'Poss', 'Gls', 'Ast']].merge(df_goleiros[['Squad', 'Save%', 'SoTA']], on='Squad', how='left')
    df_taticas = df_taticas.merge(df_diversas[['Squad', 'TklW', 'CrdY']], on='Squad', how='left')
    
    colunas_features = ['Poss', 'Gls', 'Ast', 'Save%', 'SoTA', 'TklW', 'CrdY']
    media_geral = df_taticas[colunas_features].mean().to_dict()

    def extrair_gols(df):
        df_temp = df.copy()
        gols = df_temp['Score'].astype(str).str.extract(r'(\d+)\D+(\d+)')
        df_temp['gols_mandante'] = gols[0].astype(float)
        df_temp['gols_visitante'] = gols[1].astype(float)
        return df_temp

    df_2025 = extrair_gols(df_2025).dropna(subset=['gols_mandante', 'gols_visitante'])
    df_2026 = extrair_gols(df_2026)

    def integrar_taticas_e_deltas(df_jogos):
        df = df_jogos.merge(df_taticas, left_on='Home', right_on='Squad', how='left')
        df.rename(columns={c: f"{c}_mandante" for c in colunas_features}, inplace=True)
        df.drop('Squad', axis=1, inplace=True)
        
        df = df.merge(df_taticas, left_on='Away', right_on='Squad', how='left')
        df.rename(columns={c: f"{c}_visitante" for c in colunas_features}, inplace=True)
        df.drop('Squad', axis=1, inplace=True)
        
        for c in colunas_features:
            df[f"{c}_mandante"] = df[f"{c}_mandante"].fillna(media_geral[c])
            df[f"{c}_visitante"] = df[f"{c}_visitante"].fillna(media_geral[c])
            df[f"Delta_{c}"] = df[f"{c}_mandante"] - df[f"{c}_visitante"]
            
        return df

    df_2025_full = integrar_taticas_e_deltas(df_2025)
    df_2026_full = integrar_taticas_e_deltas(df_2026)
    todos_times = sorted(list(set(df_2026_full['Home'].dropna().unique()) | set(df_2026_full['Away'].dropna().unique())))

    return df_2025_full, df_2026_full, colunas_features, todos_times

df_2025_full, df_2026_full, colunas_features, lista_times = carregar_dados()

# ==========================================
# 3. PIPELINE DE IA E ACURÁCIA (Ajuste Fino)
# ==========================================
features_modelo = [f"Delta_{c}" for c in colunas_features]

@st.cache_resource
def treinar_modelo(df_treino, features):
    X = df_treino[features]
    y = df_treino[['gols_mandante', 'gols_visitante']]
    
    modelo_base = GradientBoostingRegressor(n_estimators=300, learning_rate=0.01, max_depth=3, subsample=0.8, random_state=42)
    modelo = MultiOutputRegressor(modelo_base)
    modelo.fit(X, y)
    return modelo

modelo_gb = treinar_modelo(df_2025_full, features_modelo)

def calcular_acuracia_resultado(modelo, df_teste, features):
    if len(df_teste) == 0: return 0.0
    previsoes = modelo.predict(df_teste[features])
    acertos = 0
    total = len(df_teste)
    
    for i in range(total):
        real_m = df_teste.iloc[i]['gols_mandante']
        real_v = df_teste.iloc[i]['gols_visitante']
        diff_prev = previsoes[i][0] - previsoes[i][1]
        
        real_res = 1 if real_m > real_v else (-1 if real_v > real_m else 0)
        # Limiar otimizado para 0.35 para maximizar a assertividade
        prev_res = 1 if diff_prev > 0.35 else (-1 if diff_prev < -0.35 else 0)
        
        if real_res == prev_res: acertos += 1
            
    return (acertos / total) * 100

df_2026_realizados = df_2026_full.dropna(subset=['gols_mandante', 'gols_visitante'])
acuracia = calcular_acuracia_resultado(modelo_gb, df_2026_realizados, features_modelo)

st.metric(label="🎯 Assertividade da IA (Resultado do Confronto)", 
          value=f"{acuracia:.1f}%", 
          delta=f"Mapeado nos {len(df_2026_realizados)} jogos reais desta temporada")

# ==========================================
# 4. INTERFACE E RENDERIZAÇÃO LIVE
# ==========================================
rodadas_disponiveis = sorted(df_2026_full['Wk'].dropna().unique())
rodada_selecionada = st.selectbox("Rodada", rodadas_disponiveis, format_func=lambda x: f"Rodada {int(x)}")

df_rodada = df_2026_full[df_2026_full['Wk'] == rodada_selecionada].copy()
df_acumulado = df_2026_full[df_2026_full['Wk'] <= rodada_selecionada].copy()

aba_partidas, aba_classificacao = st.tabs(["Partidas & Análise Tática", "Classificação (Simulada)"])

with aba_partidas:
    st.markdown("<br>", unsafe_allow_html=True)
    
    for index, jogo in df_rodada.iterrows():
        X_pred = jogo[features_modelo].values.reshape(1, -1)
        previsao = modelo_gb.predict(X_pred)[0]
        
        diff_prev = previsao[0] - previsao[1]
        
        # Ajuste inteligente para o visual do placar com o limiar balanceado de 0.35
        if diff_prev > 0.35:
            gols_m_ia, gols_v_ia = max(1, int(np.round(previsao[0]))), max(0, int(np.round(previsao[1])))
        elif diff_prev < -0.35:
            gols_m_ia, gols_v_ia = max(0, int(np.round(previsao[0]))), max(1, int(np.round(previsao[1])))
        else:
            gols_empate = max(0, int(np.round((previsao[0] + previsao[1])/2)))
            gols_m_ia, gols_v_ia = gols_empate, gols_empate
        
        placar_real_html = ""
        if pd.notna(jogo['gols_mandante']) and pd.notna(jogo['gols_visitante']):
            placar_real_html = f'<div class="real-score">RESULTADO REAL: {int(jogo["gols_mandante"])} - {int(jogo["gols_visitante"])}</div>'
        
        html_card = f'<div class="sports-card">'
        html_card += f'<div class="match-status">MÉTRICA PREDITIVA IA • {jogo["Venue"]}</div>'
        html_card += placar_real_html
        html_card += '<div class="match-row">'
        html_card += f'<div class="team-name team-home">{jogo["Home"]}</div>'
        html_card += f'<div class="score-container"><div class="score-box">{gols_m_ia} - {gols_v_ia}</div></div>'
        html_card += f'<div class="team-name team-away">{jogo["Away"]}</div>'
        html_card += '</div></div>'
        
        st.markdown(html_card, unsafe_allow_html=True)

        with st.expander("Ver Análise Tática do Confronto"):
            df_comparativo = pd.DataFrame({
                'Indicador Tático': ['Gols Coletivos', 'Posse de Bola', 'Desarmes Eficientes', 'Chutes Sofridos', 'Defesas do Goleiro'],
                jogo['Home']: [f"{jogo['Gls_mandante']:.0f}", f"{jogo['Poss_mandante']:.1f}%", f"{jogo['TklW_mandante']:.0f}", f"{jogo['SoTA_mandante']:.0f}", f"{jogo['Save%_mandante']:.1f}%"],
                jogo['Away']: [f"{jogo['Gls_visitante']:.0f}", f"{jogo['Poss_visitante']:.1f}%", f"{jogo['TklW_visitante']:.0f}", f"{jogo['SoTA_visitante']:.0f}", f"{jogo['Save%_visitante']:.1f}%"]
            })
            st.dataframe(df_comparativo, hide_index=True, use_container_width=True)

with aba_classificacao:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"### Classificação Acumulada por IA (Até a {int(rodada_selecionada)}ª Rodada)")
    
    estatisticas_times = {time: {'Pts': 0, 'J': 0, 'V': 0, 'E': 0, 'D': 0, 'GP': 0, 'GC': 0, 'SG': 0} for time in lista_times}
    
    for index, row in df_acumulado.iterrows():
        mandante, visitante = row['Home'], row['Away']
        if pd.isna(mandante) or pd.isna(visitante): continue
            
        X_pred = row[features_modelo].values.reshape(1, -1)
        previsao = modelo_gb.predict(X_pred)[0]
        diff_prev = previsao[0] - previsao[1]
        
        if diff_prev > 0.35: gols_m, gols_v = max(1, int(np.round(previsao[0]))), max(0, int(np.round(previsao[1])))
        elif diff_prev < -0.35: gols_m, gols_v = max(0, int(np.round(previsao[0]))), max(1, int(np.round(previsao[1])))
        else:
            gols_empate = max(0, int(np.round((previsao[0] + previsao[1])/2)))
            gols_m, gols_v = gols_empate, gols_empate
        
        if mandante in estatisticas_times and visitante in estatisticas_times:
            estatisticas_times[mandante]['J'] += 1; estatisticas_times[visitante]['J'] += 1
            estatisticas_times[mandante]['GP'] += gols_m; estatisticas_times[mandante]['GC'] += gols_v
            estatisticas_times[visitante]['GP'] += gols_v; estatisticas_times[visitante]['GC'] += gols_m
            
            if gols_m > gols_v:
                estatisticas_times[mandante]['Pts'] += 3; estatisticas_times[mandante]['V'] += 1; estatisticas_times[visitante]['D'] += 1
            elif gols_m == gols_v:
                estatisticas_times[mandante]['Pts'] += 1; estatisticas_times[visitante]['Pts'] += 1
                estatisticas_times[mandante]['E'] += 1; estatisticas_times[visitante]['E'] += 1
            else:
                estatisticas_times[visitante]['Pts'] += 3; estatisticas_times[visitante]['V'] += 1; estatisticas_times[mandante]['D'] += 1

    for time in estatisticas_times: estatisticas_times[time]['SG'] = estatisticas_times[time]['GP'] - estatisticas_times[time]['GC']
        
    df_tabela = pd.DataFrame.from_dict(estatisticas_times, orient='index').reset_index()
    df_tabela.rename(columns={'index': 'Clube'}, inplace=True)
    df_tabela = df_tabela.sort_values(by=['Pts', 'V', 'SG', 'GP'], ascending=[False, False, False, False]).reset_index(drop=True)
    
    html_tabela = '<div class="sports-table-container"><table class="sports-table">'
    html_tabela += '<thead><tr><th class="col-pos">Pos.</th><th class="col-clube">Clube</th><th class="col-pts">Pts</th>'
    html_tabela += '<th class="col-num">J</th><th class="col-num">V</th><th class="col-num">E</th><th class="col-num">D</th>'
    html_tabela += '<th class="col-num">GP</th><th class="col-num">GC</th><th class="col-num">SG</th></tr></thead><tbody>'
    
    for i, row in df_tabela.iterrows():
        posicao = i + 1
        classe_borda = 'class="g4-mark"' if posicao <= 4 else 'class="z4-mark"' if posicao >= 17 else ''
        html_tabela += f'<tr {classe_borda}><td class="col-pos">{posicao}</td><td class="col-clube">{row["Clube"]}</td>'
        html_tabela += f'<td class="col-pts">{row["Pts"]}</td><td class="col-num">{row["J"]}</td>'
        html_tabela += f'<td class="col-num">{row["V"]}</td><td class="col-num">{row["E"]}</td>'
        html_tabela += f'<td class="col-num">{row["D"]}</td><td class="col-num">{row["GP"]}</td>'
        html_tabela += f'<td class="col-num">{row["GC"]}</td><td class="col-num">{row["SG"]}</td></tr>'
        
    html_tabela += '</tbody></table></div>'
    st.markdown(html_tabela, unsafe_allow_html=True)