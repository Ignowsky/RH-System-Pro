import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sys
import time

# --- SETUP DE ARQUITETURA ---
sys.path.append('.')
import config
from src.processing import data_manager

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="People Analytics System",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS E ESTILO ---
st.markdown("""
    <style>
    .metric-card {background-color: #f9f9f9; border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 5px;}
    h3 {color: #333;}
    .stButton>button {width: 100%; border-radius: 5px; height: 3em; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)


# --- FUNÇÕES DE CARREGAMENTO ---
@st.cache_resource
def load_model_system():
    try:
        model = joblib.load(config.MODEL_PATH)
        features = joblib.load(config.FEATURES_PATH)
        return model, features
    except FileNotFoundError:
        return None, None


def process_and_predict(df_raw, model, train_features):
    # Pipeline Modular
    df_clean = data_manager.clean_data(df_raw)
    df_processed = data_manager.enconding_data(df_clean)
    X_new = df_processed.reindex(columns=train_features, fill_value=0)
    probs = model.predict_proba(X_new)[:, 1]
    return probs


# --- SIDEBAR (Barra Lateral) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
    st.title("RH System Enterprise")
    st.markdown("---")

    # Status do Sistema
    st.success("🟢 Servidor: Online")
    st.info(f"⚙️ Modelo: v2.0 (Prod)")
    st.caption(f"Threshold Crítico: {config.THRESHOLD_ALERT * 100:.0f}%")

    st.markdown("---")

    # Botão de Reset (Simular Logout/Refresh)
    if st.button("🔄 Recarregar Dados"):
        if 'dados_rh' in st.session_state:
            del st.session_state['dados_rh']
        st.rerun()

# --- LÓGICA PRINCIPAL ---
st.title("📊 Painel de Gestão de Pessoas (ERP Integrated)")

# 1. Carregar Modelo
model, train_features = load_model_system()
if model is None:
    st.error("🚨 Erro de Conexão com Model Registry. Verifique os arquivos .pkl")
    st.stop()

# 2. Conexão com "Banco de Dados" (Simulado)
if 'dados_rh' not in st.session_state:
    st.session_state['dados_rh'] = None

# TELA DE "LOGIN/CONEXÃO"
if st.session_state['dados_rh'] is None:
    st.markdown("### 👋 Bem-vindo ao Sistema Corporativo")
    st.markdown("Clique abaixo para conectar ao Data Warehouse e carregar os dados em tempo real.")

    if st.button("🔌 CONECTAR AO SERVIDOR (DATABASE 25K)"):
        with st.spinner("Estabelecendo conexão segura com o ERP..."):
            time.sleep(1.5)  # Charme: finge que está conectando na rede

        with st.spinner("Baixando registros (Query SQL)..."):
            try:
                # AQUI É O PULO DO GATO: Carrega direto do config, sem upload
                df_temp = data_manager.load_dataset(config.NEW_DATA_FILE)
                st.session_state['dados_rh'] = df_temp
                st.success(f"Conexão estabelecida! {len(df_temp)} registros carregados.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Falha na conexão com o arquivo de dados: {e}")
                st.warning("Verifique se 'Base_Stress_25k.csv' está na pasta 'data/new/' no repositório.")

# 3. O SISTEMA (DASHBOARD)
else:
    df_input = st.session_state['dados_rh']

    # Criação das Abas
    tab1, tab2 = st.tabs(["📊 Visão Geral (Dashboard)", "🔮 Predição de Risco (IA)"])

    # ===================================================
    # ABA 1: DASHBOARD DEMOGRÁFICO
    # ===================================================
    with tab1:
        st.markdown("### 🏢 Raio-X da Organização")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Ativos", f"{len(df_input):,.0f}")

        if 'Age' in df_input.columns:
            kpi2.metric("Idade Média", f"{df_input['Age'].mean():.0f} anos")
        if 'MonthlyIncome' in df_input.columns:
            kpi3.metric("Ticket Médio (Salário)", f"R$ {df_input['MonthlyIncome'].mean():,.2f}")
        if 'YearsAtCompany' in df_input.columns:
            kpi4.metric("Avg. Tenure", f"{df_input['YearsAtCompany'].mean():.1f} anos")

        st.divider()

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            if 'Department' in df_input.columns:
                fig_dept = px.histogram(df_input, x='Department', title="Headcount por Departamento",
                                        color='Department', text_auto=True)
                st.plotly_chart(fig_dept, use_container_width=True)
        with col_g2:
            if 'OverTime' in df_input.columns:
                fig_ot = px.pie(df_input, names='OverTime', title="Monitoramento de Horas Extras",
                                color='OverTime', color_discrete_map={'Yes': 'red', 'No': '#2bd966'}, hole=0.4)
                st.plotly_chart(fig_ot, use_container_width=True)

        col_g3, col_g4 = st.columns(2)
        with col_g3:
            if 'MonthlyIncome' in df_input.columns:
                fig_sal = px.box(df_input, x='Department', y='MonthlyIncome', color='Department',
                                 title="Faixa Salarial por Área")
                st.plotly_chart(fig_sal, use_container_width=True)
        with col_g4:
            if 'Age' in df_input.columns:
                fig_age = px.histogram(df_input, x='Age', nbins=20, title="Demografia (Idade)",
                                       color_discrete_sequence=['#3366cc'])
                st.plotly_chart(fig_age, use_container_width=True)

    # ===================================================
    # ABA 2: PREDIÇÃO (IA)
    # ===================================================
    with tab2:
        st.markdown("### 🤖 Motor de Inteligência Preditiva")

        if st.button("🧠 EXECUTAR ALGORITMO DE RETENÇÃO"):
            with st.spinner("Processando rede neural..."):
                probs = process_and_predict(df_input, model, train_features)

                df_view = df_input.copy()
                df_view['Probabilidade'] = probs
                df_view['Risco'] = df_view['Probabilidade'].apply(
                    lambda x: '🔴 CRÍTICO' if x >= 0.70 else ('🟡 ALERTA' if x >= config.THRESHOLD_ALERT else '🟢 BAIXO')
                )

                # KPIs
                total = len(df_view)
                criticos = len(df_view[df_view['Risco'] == '🔴 CRÍTICO'])
                alertas = len(df_view[df_view['Risco'] == '🟡 ALERTA'])

                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Risco Crítico", criticos, delta_color="inverse")
                c2.metric("🟡 Alerta", alertas, delta_color="inverse")
                c3.metric("🟢 Estável", total - (criticos + alertas))

                st.progress((criticos + alertas) / total, text="Índice de Risco Global")

                # Gráficos
                row_r1, row_r2 = st.columns(2)
                with row_r1:
                    fig_risk_bar = px.histogram(df_view, x='Department', color='Risco', barmode='group',
                                                title="Risco por Departamento",
                                                color_discrete_map={'🔴 CRÍTICO': 'red', '🟡 ALERTA': 'orange',
                                                                    '🟢 BAIXO': 'green'})
                    st.plotly_chart(fig_risk_bar, use_container_width=True)
                with row_r2:
                    if 'MonthlyIncome' in df_view.columns and 'Age' in df_view.columns:
                        fig_scatter = px.scatter(df_view, x='Age', y='MonthlyIncome', color='Risco',
                                                 title="Matriz de Risco: Idade x Salário", opacity=0.6,
                                                 color_discrete_map={'🔴 CRÍTICO': 'red', '🟡 ALERTA': 'orange',
                                                                     '🟢 BAIXO': 'green'})
                        st.plotly_chart(fig_scatter, use_container_width=True)

                # Tabela
                st.subheader("📋 Relatório Analítico")
                filtro = st.toggle("Ver apenas Colaboradores em Risco", value=True)
                df_table = df_view.sort_values('Probabilidade', ascending=False)

                if filtro:
                    df_table = df_table[df_table['Probabilidade'] >= config.THRESHOLD_ALERT]

                cols_view = ['EmployeeNumber', 'Name', 'Age', 'Department', 'OverTime', 'MonthlyIncome', 'Risco',
                             'Probabilidade']
                cols_final = [c for c in cols_view if c in df_table.columns]

                st.dataframe(
                    df_table[cols_final].style.format({'Probabilidade': '{:.1%}', 'MonthlyIncome': 'R$ {:,.2f}'})
                    .applymap(lambda v: 'color: red; font-weight: bold;' if v == '🔴 CRÍTICO' else None),
                    use_container_width=True
                )