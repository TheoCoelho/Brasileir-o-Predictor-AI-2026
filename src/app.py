import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
import numpy as np

# ==========================================
# 1. CONFIGURAÇÕES E ESTILO
# ==========================================
st.set_page_config(page_title="Brasileirão Pro Analytics", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {display: none;}

        /* ── BASE ── */
        .stApp { background-color: #080C10; color: #C8D6E5; font-family: 'Inter', sans-serif; }

        /* ── CABEÇALHO ── */
        .cabecalho { padding: 28px 0 20px 0; border-bottom: 1px solid #1A2333; margin-bottom: 24px; }
        .titulo-principal { font-family: 'Inter', sans-serif; font-weight: 800; font-size: 28px; color: #F0F6FF; margin: 0; letter-spacing: -0.5px; }
        .titulo-destaque { color: #3D8EFF; }
        .subtitulo { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #4A6080; margin-top: 6px; letter-spacing: 1.5px; text-transform: uppercase; }

        /* ── CARTÕES DE PARTIDA ── */
        .card-partida { background: #0D1520; border: 1px solid #1A2333; border-radius: 10px; padding: 0; margin-bottom: 8px; overflow: hidden; transition: border-color 0.2s ease, transform 0.15s ease; }
        .card-partida:hover { border-color: #2A4A7F; transform: translateY(-1px); }
        .card-topo { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px 10px 16px; background: #0A1018; border-bottom: 1px solid #1A2333; }
        .info-estadio { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #3D5470; text-transform: uppercase; letter-spacing: 0.8px; }
        .badge-ia { background: rgba(61, 142, 255, 0.12); color: #3D8EFF; font-family: 'JetBrains Mono', monospace; font-size: 9px; font-weight: 700; padding: 3px 8px; border-radius: 20px; border: 1px solid rgba(61, 142, 255, 0.25); letter-spacing: 0.5px; }
        .card-corpo { padding: 20px 20px 16px 20px; }
        .linha-times { display: flex; justify-content: space-between; align-items: center; gap: 12px; }
        .time-bloco { flex: 1; display: flex; flex-direction: column; gap: 4px; }
        .time-casa { align-items: flex-end; text-align: right; }
        .time-fora { align-items: flex-start; text-align: left; }
        .nome-time { font-weight: 700; font-size: 15px; color: #E8F0FC; line-height: 1.2; }
        .label-time { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #2D4560; text-transform: uppercase; letter-spacing: 1px; }
        .placar-caixa { background: #080C10; border: 1px solid #1E2D42; border-radius: 8px; padding: 10px 18px; font-family: 'JetBrains Mono', monospace; font-size: 24px; font-weight: 700; color: #F0F6FF; text-align: center; min-width: 88px; letter-spacing: 2px; flex-shrink: 0; }

        /* ── BARRA DE PROBABILIDADE ── */
        .barra-prob-container { margin-top: 14px; }
        .barra-prob-labels { display: flex; justify-content: space-between; font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #3D5470; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
        .barra-prob-track { background: #0A1018; border-radius: 3px; height: 4px; width: 100%; overflow: hidden; display: flex; }
        .barra-seg-casa { background: #3D8EFF; height: 100%; border-radius: 3px 0 0 3px; }
        .barra-seg-empate { background: #4A6080; height: 100%; }
        .barra-seg-fora { background: #00C853; height: 100%; border-radius: 0 3px 3px 0; }

        /* ── RESULTADO REAL ── */
        .resultado-real { margin-top: 12px; padding: 8px 12px; background: rgba(245, 101, 82, 0.06); border: 1px solid rgba(245, 101, 82, 0.15); border-radius: 6px; font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #F56552; text-align: center; letter-spacing: 0.5px; }
        .resultado-real-correto { background: rgba(0, 200, 83, 0.06); border-color: rgba(0, 200, 83, 0.15); color: #00C853; }

        /* ── TABELA DE CLASSIFICAÇÃO ── */
        .tabela-classificacao { width: 100%; border-collapse: collapse; font-family: 'Inter', sans-serif; }
        .tabela-classificacao thead tr { background: #0A1018; border-bottom: 2px solid #1A2333; }
        .tabela-classificacao th { font-family: 'JetBrains Mono', monospace; font-size: 10px; font-weight: 500; color: #3D5470; padding: 12px 14px; text-align: center; text-transform: uppercase; letter-spacing: 1px; }
        .tabela-classificacao th.col-clube-th { text-align: left; }
        .tabela-classificacao td { font-size: 13px; color: #A8BACA; padding: 11px 14px; text-align: center; border-bottom: 1px solid #0F1A27; transition: background 0.15s; }
        .tabela-classificacao tbody tr:hover td { background: #0D1520; }
        .col-pos { font-family: 'JetBrains Mono', monospace; color: #3D5470 !important; font-size: 12px !important; width: 36px; }
        .col-clube { text-align: left !important; font-weight: 600 !important; color: #E0EAF8 !important; }
        .col-pts { font-family: 'JetBrains Mono', monospace; font-weight: 700 !important; color: #F0F6FF !important; font-size: 15px !important; }

        /* ── MARCAÇÕES DE ZONA E ABAS ── */
        .zona-libertadores td:first-child { border-left: 3px solid #3D8EFF; }
        .zona-sulamericana td:first-child { border-left: 3px solid #00C853; }
        .zona-rebaixamento td:first-child { border-left: 3px solid #F56552; }
        .legenda-zona { display: flex; gap: 20px; padding: 12px 0 6px 0; margin-top: 12px; }
        .legenda-item { display: flex; align-items: center; gap: 6px; font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #3D5470; text-transform: uppercase; letter-spacing: 0.5px; }
        .legenda-cor { width: 10px; height: 10px; border-radius: 2px; }
        .stTabs [data-baseweb="tab-list"] { gap: 0; background: #0A1018; border-radius: 8px; padding: 4px; border: 1px solid #1A2333; }
        .stTabs [data-baseweb="tab"] { height: 38px; font-family: 'Inter', sans-serif; font-size: 13px; font-weight: 500; background-color: transparent; border-radius: 6px; color: #4A6080; border: none; padding: 0 20px; letter-spacing: 0.2px; }
        .stTabs [aria-selected="true"] { color: #F0F6FF; background-color: #1A2A40 !important; font-weight: 600; }
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

        /* ── MÉTRICAS E SELECTBOX ── */
        [data-testid="stMetric"] { background: #0D1520; border: 1px solid #1A2333; border-radius: 8px; padding: 14px 16px !important; }
        [data-testid="stMetricLabel"], .stSelectbox label { font-family: 'JetBrains Mono', monospace; font-size: 10px !important; color: #3D5470 !important; text-transform: uppercase; letter-spacing: 1px; }
        [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 28px !important; font-weight: 700 !important; color: #3D8EFF !important; }
        [data-testid="stMetricDelta"] { font-size: 11px !important; color: #00C853 !important; }
        .divisor { border: none; border-top: 1px solid #1A2333; margin: 20px 0; }
        .status-sistema { font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #2D4560; text-align: right; padding-top: 8px; line-height: 1.8; }
        .status-online { color: #00C853; }
        
        /* Ajuste fino para os expanders */
        .streamlit-expanderHeader { font-family: 'JetBrains Mono', monospace; font-size: 11px; color: #3D8EFF; }
    </style>
""", unsafe_allow_html=True)

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
        st.error(f"Erro Crítico de I/O: Falha ao ler base de dados em {data_dir}.")
        st.stop()

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

    for col in ['Poss', 'Gls', 'Ast']: df_elenco[col] = pd.to_numeric(df_elenco[col], errors='coerce').fillna(0)
    for col in ['Save%', 'SoTA']: df_goleiros[col] = pd.to_numeric(df_goleiros[col], errors='coerce').fillna(50)
    for col in ['TklW', 'CrdY']: df_diversas[col] = pd.to_numeric(df_diversas[col], errors='coerce').fillna(0)

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
# 3. PIPELINE DE IA E ACURÁCIA
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
        real_m, real_v = df_teste.iloc[i]['gols_mandante'], df_teste.iloc[i]['gols_visitante']
        diff_prev = previsoes[i][0] - previsoes[i][1]
        real_res = 1 if real_m > real_v else (-1 if real_v > real_m else 0)
        prev_res = 1 if diff_prev > 0.35 else (-1 if diff_prev < -0.35 else 0)
        if real_res == prev_res: acertos += 1
    return (acertos / total) * 100

df_2026_realizados = df_2026_full.dropna(subset=['gols_mandante', 'gols_visitante'])
acuracia = calcular_acuracia_resultado(modelo_gb, df_2026_realizados, features_modelo)

# ==========================================
# 4. CABEÇALHO E BARRA DE CONTROLES
# ==========================================
st.markdown("""
    <div class="cabecalho">
        <h1 class="titulo-principal">Brasileirão <span class="titulo-destaque">Pro Analytics</span></h1>
        <p class="subtitulo">Previsão de Partidas com IA · Simulador Tático · Modelo: Gradient Boosting</p>
    </div>
""", unsafe_allow_html=True)

col_controle, col_metrica, col_status = st.columns([1.5, 2, 1])

with col_controle:
    rodadas_disponiveis = sorted(df_2026_full['Wk'].dropna().unique())
    rodada_selecionada = st.selectbox("Rodada", rodadas_disponiveis, format_func=lambda x: f"Rodada {int(x)}")

with col_metrica:
    st.metric(
        label="Acurácia Global (1X2)",
        value=f"{acuracia:.1f}%",
        delta=f"{len(df_2026_realizados)} partidas validadas"
    )

with col_status:
    st.markdown(f"""
        <div class="status-sistema">
            SISTEMA <span class="status-online">● ONLINE</span><br>
            MODELO · GBR v1.0<br>
            DADOS · 2025–2026
        </div>
    """, unsafe_allow_html=True)

st.markdown("<hr class='divisor'>", unsafe_allow_html=True)

df_rodada = df_2026_full[df_2026_full['Wk'] == rodada_selecionada].copy()
df_acumulado = df_2026_full[df_2026_full['Wk'] <= rodada_selecionada].copy()

# ==========================================
# 5. ÁREA PRINCIPAL (ABAS) E ESTATÍSTICAS
# ==========================================
aba_partidas, aba_classificacao = st.tabs(["⚽  Partidas da Rodada", "📊  Tabela de Classificação"])

with aba_partidas:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for index, jogo in df_rodada.iterrows():
        X_pred = jogo[features_modelo].values.reshape(1, -1)
        previsao = modelo_gb.predict(X_pred)[0]
        diff_prev = previsao[0] - previsao[1]

        if diff_prev > 0.35:
            gols_m_ia, gols_v_ia = max(1, int(np.round(previsao[0]))), max(0, int(np.round(previsao[1])))
        elif diff_prev < -0.35:
            gols_m_ia, gols_v_ia = max(0, int(np.round(previsao[0]))), max(1, int(np.round(previsao[1])))
        else:
            gols_empate = max(0, int(np.round((previsao[0] + previsao[1]) / 2)))
            gols_m_ia, gols_v_ia = gols_empate, gols_empate

        # Calcula barras de probabilidade visual (baseado na diferença prevista)
        forca_casa = max(0.05, min(0.90, 0.5 + diff_prev * 0.18))
        forca_fora = max(0.05, min(0.90, 0.5 - diff_prev * 0.18))
        forca_empate = max(0.05, 1.0 - forca_casa - forca_fora)
        total_forca = forca_casa + forca_empate + forca_fora
        pct_casa = (forca_casa / total_forca) * 100
        pct_empate = (forca_empate / total_forca) * 100
        pct_fora = (forca_fora / total_forca) * 100

        resultado_real_html = ""
        if pd.notna(jogo['gols_mandante']) and pd.notna(jogo['gols_visitante']):
            gm_real = int(jogo['gols_mandante'])
            gv_real = int(jogo['gols_visitante'])
            diff_real = gm_real - gv_real
            prev_res = 1 if diff_prev > 0.35 else (-1 if diff_prev < -0.35 else 0)
            real_res = 1 if diff_real > 0 else (-1 if diff_real < 0 else 0)
            classe_resultado = "resultado-real-correto" if prev_res == real_res else "resultado-real"
            resultado_real_html = f'<div class="{classe_resultado}">Resultado Real: {gm_real} – {gv_real}</div>'

        # Construção da String HTML de forma segura (sem recuos falsos do Markdown)
        html_card = '<div class="card-partida">'
        html_card += f'<div class="card-topo"><span class="info-estadio">🏟 {jogo["Venue"]}</span><span class="badge-ia">IA · GBR</span></div>'
        html_card += '<div class="card-corpo">'
        html_card += '<div class="linha-times">'
        html_card += f'<div class="time-bloco time-casa"><span class="label-time">Mandante</span><span class="nome-time">{jogo["Home"]}</span></div>'
        html_card += f'<div class="placar-caixa">{gols_m_ia} – {gols_v_ia}</div>'
        html_card += f'<div class="time-bloco time-fora"><span class="label-time">Visitante</span><span class="nome-time">{jogo["Away"]}</span></div>'
        html_card += '</div>'
        
        html_card += '<div class="barra-prob-container">'
        html_card += f'<div class="barra-prob-labels"><span>{pct_casa:.0f}%</span><span>Empate {pct_empate:.0f}%</span><span>{pct_fora:.0f}%</span></div>'
        html_card += '<div class="barra-prob-track">'
        html_card += f'<div class="barra-seg-casa" style="width:{pct_casa:.1f}%"></div>'
        html_card += f'<div class="barra-seg-empate" style="width:{pct_empate:.1f}%"></div>'
        html_card += f'<div class="barra-seg-fora" style="width:{pct_fora:.1f}%"></div>'
        html_card += '</div></div>'
        
        html_card += resultado_real_html
        html_card += '</div></div>'

        # Dados Táticos para a tabela Expander
        df_comparativo = pd.DataFrame({
            'Indicador': ['Gols Pró (Poder de Fogo)', 'Posse de Bola', 'Desarmes (Eficiência Defensiva)', 'Chutes Sofridos no Alvo', 'Defesas do Goleiro (%)'],
            jogo['Home']: [f"{jogo['Gls_mandante']:.0f}", f"{jogo['Poss_mandante']:.1f}%", f"{jogo['TklW_mandante']:.0f}", f"{jogo['SoTA_mandante']:.0f}", f"{jogo['Save%_mandante']:.1f}%"],
            jogo['Away']: [f"{jogo['Gls_visitante']:.0f}", f"{jogo['Poss_visitante']:.1f}%", f"{jogo['TklW_visitante']:.0f}", f"{jogo['SoTA_visitante']:.0f}", f"{jogo['Save%_visitante']:.1f}%"]
        })

        if index % 2 == 0:
            with col1:
                st.markdown(html_card, unsafe_allow_html=True)
                with st.expander("📊 Ver Dados Táticos"):
                    st.dataframe(df_comparativo, hide_index=True, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)
        else:
            with col2:
                st.markdown(html_card, unsafe_allow_html=True)
                with st.expander("📊 Ver Dados Táticos"):
                    st.dataframe(df_comparativo, hide_index=True, use_container_width=True)
                st.markdown("<br>", unsafe_allow_html=True)

with aba_classificacao:
    st.markdown("<br>", unsafe_allow_html=True)

    estatisticas_times = {time: {'Pts': 0, 'J': 0, 'V': 0, 'E': 0, 'D': 0, 'GP': 0, 'GC': 0, 'SG': 0} for time in lista_times}

    for index, row in df_acumulado.iterrows():
        mandante, visitante = row['Home'], row['Away']
        if pd.isna(mandante) or pd.isna(visitante): continue

        X_pred = row[features_modelo].values.reshape(1, -1)
        previsao = modelo_gb.predict(X_pred)[0]
        diff_prev = previsao[0] - previsao[1]

        if diff_prev > 0.35:
            gols_m, gols_v = max(1, int(np.round(previsao[0]))), max(0, int(np.round(previsao[1])))
        elif diff_prev < -0.35:
            gols_m, gols_v = max(0, int(np.round(previsao[0]))), max(1, int(np.round(previsao[1])))
        else:
            gols_empate = max(0, int(np.round((previsao[0] + previsao[1]) / 2)))
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

    for time in estatisticas_times:
        estatisticas_times[time]['SG'] = estatisticas_times[time]['GP'] - estatisticas_times[time]['GC']

    df_tabela = pd.DataFrame.from_dict(estatisticas_times, orient='index').reset_index()
    df_tabela.rename(columns={'index': 'Clube'}, inplace=True)
    df_tabela = df_tabela.sort_values(by=['Pts', 'V', 'SG', 'GP'], ascending=[False, False, False, False]).reset_index(drop=True)

    # Legenda de zonas
    st.markdown("""
        <div class="legenda-zona">
            <div class="legenda-item">
                <div class="legenda-cor" style="background:#3D8EFF;"></div>
                <span>Libertadores (1–6)</span>
            </div>
            <div class="legenda-item">
                <div class="legenda-cor" style="background:#00C853;"></div>
                <span>Sul-Americana (7–12)</span>
            </div>
            <div class="legenda-item">
                <div class="legenda-cor" style="background:#F56552;"></div>
                <span>Rebaixamento (17–20)</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    html_tabela = '<table class="tabela-classificacao">'
    html_tabela += '<thead><tr>'
    html_tabela += '<th>Pos</th>'
    html_tabela += '<th class="col-clube-th">Clube</th>'
    html_tabela += '<th>Pts</th>'
    html_tabela += '<th>J</th><th>V</th><th>E</th><th>D</th>'
    html_tabela += '<th>GP</th><th>GC</th><th>SG</th>'
    html_tabela += '</tr></thead><tbody>'

    for i, row in df_tabela.iterrows():
        posicao = i + 1
        if posicao <= 6:
            classe_linha = 'class="zona-libertadores"'
        elif posicao <= 12:
            classe_linha = 'class="zona-sulamericana"'
        elif posicao >= 17:
            classe_linha = 'class="zona-rebaixamento"'
        else:
            classe_linha = ''

        sg_str = f"+{row['SG']}" if row['SG'] > 0 else str(row['SG'])

        html_tabela += f'<tr {classe_linha}>'
        html_tabela += f'<td class="col-pos">{posicao}</td>'
        html_tabela += f'<td class="col-clube">{row["Clube"]}</td>'
        html_tabela += f'<td class="col-pts">{row["Pts"]}</td>'
        html_tabela += f'<td>{row["J"]}</td>'
        html_tabela += f'<td>{row["V"]}</td>'
        html_tabela += f'<td>{row["E"]}</td>'
        html_tabela += f'<td>{row["D"]}</td>'
        html_tabela += f'<td>{row["GP"]}</td>'
        html_tabela += f'<td>{row["GC"]}</td>'
        html_tabela += f'<td>{sg_str}</td>'
        html_tabela += '</tr>'

    html_tabela += '</tbody></table>'
    st.markdown(html_tabela, unsafe_allow_html=True)