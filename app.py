import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import sys

# --- SETUP DE ARQUITETURA ---
sys.path.append('.')
import config
from src.processing import data_manager

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="People Analytics System",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS E ESTILO ---
st.markdown("""
    <style>
    .metric-card {background-color: #f9f9f9; border-left: 5px solid #ff4b4b; padding: 15px; border-radius: 5px;}
    h3 {color: #333;}
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


# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3079/3079165.png", width=80)
    st.title("RH System Pro")
    st.caption("People Analytics & AI")
    st.markdown("---")

    modo = st.radio("Fonte de Dados:", ["Upload CSV", "Base Stress Test (25k)"])
    st.info(f"⚙️ Modelo Calibrado: Threshold {config.THRESHOLD_ALERT * 100:.0f}%")

# --- LÓGICA PRINCIPAL ---
st.title("📊 Painel de Gestão de Pessoas e Retenção")

# 1. Carregar Modelo
model, train_features = load_model_system()
if model is None:
    st.error("🚨 Modelo não encontrado. Rode `src/train/train_pipeline.py`.")
    st.stop()

# 2. Carregar Dados
df_input = None
if modo == "Upload CSV":
    uploaded_file = st.file_uploader("Carregue a base atual (CSV)", type="csv")
    if uploaded_file:
        df_input = data_manager.load_data(uploaded_file)
else:
    if st.button("⚡ Carregar Dados do ERP (Simulação 25k)"):
        try:
            df_input = data_manager.load_data(config.NEW_DATA_FILE)
            st.toast(f"Dados carregados: {len(df_input)} registros", icon="✅")
        except:
            st.error("Erro ao carregar base de teste.")

# SE TIVER DADOS, MOSTRA O SISTEMA
if df_input is not None:

    # Criação das Abas
    tab1, tab2 = st.tabs(["📊 Visão Geral (Dashboard)", "🔮 Predição de Risco (IA)"])

    # ===================================================
    # ABA 1: DASHBOARD DEMOGRÁFICO & OPERACIONAL
    # ===================================================
    with tab1:
        st.markdown("### 🏢 Raio-X da Organização")

        # Métricas Gerais
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Colaboradores", len(df_input))

        if 'Age' in df_input.columns:
            kpi2.metric("Idade Média", f"{df_input['Age'].mean():.0f} anos")

        if 'MonthlyIncome' in df_input.columns:
            kpi3.metric("Média Salarial", f"R$ {df_input['MonthlyIncome'].mean():,.2f}")

        if 'YearsAtCompany' in df_input.columns:
            kpi4.metric("Tempo Médio de Casa", f"{df_input['YearsAtCompany'].mean():.1f} anos")

        st.divider()

        # Linha 1: Departamentos e Hora Extra
        col_g1, col_g2 = st.columns(2)

        with col_g1:
            if 'Department' in df_input.columns:
                fig_dept = px.histogram(df_input, x='Department', title="Distribuição por Departamento",
                                        color='Department', text_auto=True)
                st.plotly_chart(fig_dept, use_container_width=True)

        with col_g2:
            if 'OverTime' in df_input.columns:
                # Gráfico de Pizza para Hora Extra
                fig_ot = px.pie(df_input, names='OverTime', title="Proporção de Hora Extra (Burnout Risk)",
                                color='OverTime', color_discrete_map={'Yes': 'red', 'No': '#2bd966'}, hole=0.4)
                st.plotly_chart(fig_ot, use_container_width=True)

        # Linha 2: Análise Salarial e Idade
        col_g3, col_g4 = st.columns(2)

        with col_g3:
            if 'MonthlyIncome' in df_input.columns and 'Department' in df_input.columns:
                fig_sal = px.box(df_input, x='Department', y='MonthlyIncome', color='Department',
                                 title="Distribuição Salarial por Área (Boxplot)")
                st.plotly_chart(fig_sal, use_container_width=True)

        with col_g4:
            if 'Age' in df_input.columns:
                fig_age = px.histogram(df_input, x='Age', nbins=20, title="Histograma de Idades",
                                       color_discrete_sequence=['#3366cc'])
                fig_age.update_layout(bargap=0.1)
                st.plotly_chart(fig_age, use_container_width=True)

    # ===================================================
    # ABA 2: PREDIÇÃO DE RISCO (IA)
    # ===================================================
    with tab2:
        st.markdown("### 🤖 Inteligência Artificial - Previsão de Turnover")

        if st.button("🧠 Rodar Modelo Preditivo"):
            with st.spinner("Analisando padrões comportamentais..."):

                # Executa o Pipeline Modular
                probs = process_and_predict(df_input, model, train_features)

                # Prepara visualização
                df_view = df_input.copy()
                df_view['Probabilidade'] = probs
                df_view['Risco'] = df_view['Probabilidade'].apply(
                    lambda x: '🔴 CRÍTICO' if x >= 0.70 else ('🟡 ALERTA' if x >= config.THRESHOLD_ALERT else '🟢 BAIXO')
                )

                # KPIs de Risco
                total = len(df_view)
                criticos = len(df_view[df_view['Risco'] == '🔴 CRÍTICO'])
                alertas = len(df_view[df_view['Risco'] == '🟡 ALERTA'])

                # Exibição
                c1, c2, c3 = st.columns(3)
                c1.metric("🔴 Risco Crítico (>70%)", criticos, delta_color="inverse")
                c2.metric("🟡 Alerta Preventivo (>30%)", alertas, delta_color="inverse")
                c3.metric("🟢 Retenção Provável", total - (criticos + alertas))

                st.progress((criticos + alertas) / total, text="Nível de Risco da Folha")

                # Gráficos de Risco
                row_r1, row_r2 = st.columns(2)

                with row_r1:
                    fig_risk_bar = px.histogram(df_view, x='Department', color='Risco', barmode='group',
                                                title="Risco por Departamento",
                                                color_discrete_map={'🔴 CRÍTICO': 'red', '🟡 ALERTA': 'orange',
                                                                    '🟢 BAIXO': 'green'})
                    st.plotly_chart(fig_risk_bar, use_container_width=True)

                with row_r2:
                    # Scatter plot: Salário vs Idade colorido por Risco
                    if 'MonthlyIncome' in df_view.columns and 'Age' in df_view.columns:
                        fig_scatter = px.scatter(df_view, x='Age', y='MonthlyIncome', color='Risco',
                                                 title="Risco: Idade vs Salário", opacity=0.6,
                                                 color_discrete_map={'🔴 CRÍTICO': 'red', '🟡 ALERTA': 'orange',
                                                                     '🟢 BAIXO': 'green'})
                        st.plotly_chart(fig_scatter, use_container_width=True)

                # Tabela Final
                st.subheader("📋 Plano de Ação (Lista de Prioridade)")

                filtro = st.toggle("Filtrar apenas Alto Risco", value=True)
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

else:
    st.info("👆 Selecione uma fonte de dados na barra lateral para iniciar.")