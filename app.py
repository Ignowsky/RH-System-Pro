import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import time
import numpy as np
import pathlib

# ==========================================
# 1. BRAND SYSTEM (Full People Analytics)
# ==========================================
BRAND_COLORS = {
    'primary': '#0E1F26',  # Azul Petróleo
    'accent': '#2FA4A9',  # Verde Técnico
    'danger': '#D9534F',  # Vermelho Suave
    'warning': '#F59E0B',  # Amarelo/Laranja
    'neutral': '#8A8F95',  # Cinza
    'light': '#E3E6E8'  # Cinza Claro
}

# --- MOCKS PARA ROBUSTEZ ---
try:
    import config
    from src.processing import data_manager
except ImportError:
    class ConfigMock:
        MODEL_PATH = 'turnover_model.pkl'
        FEATURES_PATH = 'model_columns.pkl'
        NEW_DATA_FILE = 'Base_Geral_50k.csv'
        THRESHOLD_ALERT = 0.40


    config = ConfigMock()


    class DataManagerMock:
        def clean_data(self, df): return df

        def load_data(self, path): return pd.read_csv(path)


    data_manager = DataManagerMock()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="RH System Pro | People Analytics",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. CSS ENGINE
# ==========================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: {BRAND_COLORS['primary']};
    }}
    [data-testid="stSidebar"] * {{
        color: #F7F9FA !important;
    }}
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] .stSelectbox div {{
        color: {BRAND_COLORS['primary']} !important;
        background-color: white !important;
    }}

    /* CARDS KPI */
    .metric-card {{
        background-color: #FFFFFF;
        border-left: 4px solid {BRAND_COLORS['accent']};
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }}
    .metric-label {{
        font-size: 11px;
        font-weight: 700;
        color: {BRAND_COLORS['neutral']};
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    .metric-value {{
        font-size: 26px;
        font-weight: 700;
        color: {BRAND_COLORS['primary']};
        margin-top: 5px;
    }}

    /* BOTÕES */
    .stButton>button {{
        background-color: {BRAND_COLORS['accent']};
        color: white !important;
        border-radius: 4px;
        font-weight: 600;
        text-transform: uppercase;
        height: 3em;
        border: none;
        transition: all 0.2s;
    }}
    .stButton>button:hover {{
        background-color: #248c91;
        transform: translateY(-2px);
    }}

    /* TITULOS */
    h1, h2, h3, h4 {{
        color: {BRAND_COLORS['primary']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# --- HELPER: CARD VISUAL ---
def kpi_card(label, value, delta=None, color="neutral"):
    delta_html = ""
    if delta:
        c = BRAND_COLORS['neutral']
        if color == "pos":
            c = BRAND_COLORS['accent']
        elif color == "neg":
            c = BRAND_COLORS['danger']
        elif color == "warn":
            c = BRAND_COLORS['warning']  # Adicionado suporte para Amarelo
        delta_html = f"<div style='font-size:12px; font-weight:600; color:{c}; margin-top:4px;'>{delta}</div>"

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# --- MODELO ---
@st.cache_resource
def load_model_system():
    try:
        model = joblib.load(config.MODEL_PATH)
        try:
            features = joblib.load(config.FEATURES_PATH)
        except:
            features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        return model, features
    except FileNotFoundError:
        return None, None


def process_and_predict(df_raw, model, train_features):
    cat_cols = df_raw.select_dtypes(include=['object']).columns
    df_processed = pd.get_dummies(df_raw, columns=cat_cols, drop_first=True)
    if train_features:
        for col in train_features:
            if col not in df_processed.columns: df_processed[col] = 0
        df_processed = df_processed[train_features]
    return model.predict_proba(df_processed)[:, 1]


# ==========================================
# INTERFACE
# ==========================================

with st.sidebar:
    st.markdown("### RH System Pro")
    st.caption("PEOPLE ANALYTICS ENTERPRISE")
    st.success("🟢 ONLINE")

    st.markdown("### ⚙️ CALIBRAGEM")
    threshold_user = st.slider("Sensibilidade", 0.0, 1.0, 0.40, 0.05)

    st.divider()
    if st.button("🔄 RECARREGAR"):
        st.session_state.clear()
        st.rerun()

st.markdown("## Painel de Estratégia de Talentos")
st.markdown("Visão integrada de dados e decisões humanas.")
st.divider()

# CARGA DE DADOS
model, train_features = load_model_system()

if 'dados_rh' not in st.session_state or st.session_state['dados_rh'] is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.info("ℹ️ Conecte-se ao Data Warehouse.")
        if st.button("CONECTAR BASE DE DADOS", use_container_width=True):
            try:
                path = str(config.NEW_DATA_FILE)
                if not path.startswith('data') and not pathlib.Path(path).exists(): path = f"data/{path}"
                st.session_state['dados_rh'] = pd.read_csv(path)
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")
else:
    df_input = st.session_state['dados_rh']

    # ---------------- ABAS ----------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 VISÃO GERAL", "🔮 PREDIÇÃO & DIAGNÓSTICO", "🧪 SIMULADOR", "💰 ROI"])

    # =========================================
    # ABA 1: VISÃO GERAL
    # =========================================
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Headcount", f"{len(df_input):,.0f}", "+120", "pos")
        with c2: kpi_card("Idade Média", f"{df_input['Age'].mean():.0f} Anos")
        with c3: kpi_card("Salário Médio", f"R$ {df_input['MonthlyIncome'].mean():,.0f}", "+4%", "neg")
        with c4: kpi_card("Tempo de Casa", f"{df_input['YearsAtCompany'].mean():.1f} Anos")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### Distribuição por Departamento")
            fig = px.histogram(df_input, x='Department', color='Department',
                               color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['accent'], '#4a7c94'])
            fig.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            st.markdown("##### Gênero")
            fig2 = px.pie(df_input, names='Gender', hole=0.6,
                          color_discrete_sequence=[BRAND_COLORS['accent'], BRAND_COLORS['light']])
            fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig2, use_container_width=True)

        g3, g4 = st.columns(2)
        with g3:
            st.markdown("##### Demografia (Histograma de Idade)")
            fig_age = px.histogram(df_input, x='Age', nbins=20,
                                   color_discrete_sequence=[BRAND_COLORS['primary']])
            fig_age.update_layout(bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_age, use_container_width=True)
        with g4:
            st.markdown("##### Faixa Salarial por Departamento (Boxplot)")
            fig_box = px.box(df_input, x='Department', y='MonthlyIncome', color='Department',
                             color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['accent'], '#4a7c94'])
            fig_box.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("##### Correlação: Idade vs Renda")
        fig_scat = px.scatter(df_input.sample(min(2000, len(df_input))), x='Age', y='MonthlyIncome',
                              color='Department', size='TotalWorkingYears',
                              color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['accent'], '#F59E0B'])
        fig_scat.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_scat, use_container_width=True)

    # =========================================
    # ABA 2: PREDIÇÃO (KPIS ADICIONADOS)
    # =========================================
    with tab2:
        c_left, c_right = st.columns([3, 1])
        with c_left:
            st.markdown("##### Diagnóstico de Risco (XGBoost)")
            st.caption(f"Analisando padrões ocultos. Threshold: {threshold_user:.0%}")
        with c_right:
            if st.button("⚡ RODAR MODELO", use_container_width=True):
                probs = process_and_predict(df_input, model, train_features)
                df_res = df_input.copy()
                df_res['Probabilidade'] = probs
                st.session_state['resultado_ia'] = df_res

        if 'resultado_ia' in st.session_state:
            df_view = st.session_state['resultado_ia']
            df_view['Risco'] = df_view['Probabilidade'].apply(
                lambda x: 'CRÍTICO' if x >= 0.7 else ('ALERTA' if x >= threshold_user else 'BAIXO'))

            # --- KPIS DE RISCO (NOVO, CONFORME SOLICITADO) ---
            n_crit = len(df_view[df_view['Risco'] == 'CRÍTICO'])
            n_alert = len(df_view[df_view['Risco'] == 'ALERTA'])
            n_safe = len(df_view[df_view['Risco'] == 'BAIXO'])

            k1, k2, k3 = st.columns(3)
            with k1:
                kpi_card("Alto Risco (Crítico)", f"{n_crit}", "Ação Imediata", "neg")
            with k2:
                kpi_card("Estado de Alerta", f"{n_alert}", "Monitorar", "warn")
            with k3:
                kpi_card("Zona Segura", f"{n_safe}", "Estável", "pos")
            # ------------------------------------------------

            # 1. Feature Importance & Setor
            st.markdown("---")
            col_feat, col_risk = st.columns(2)

            with col_feat:
                st.markdown("##### 🔍 Top Fatores de Influência")
                if hasattr(model, 'feature_importances_'):
                    feat_imp = pd.DataFrame({
                        'Fator': train_features,
                        'Peso': model.feature_importances_
                    }).sort_values('Peso', ascending=True).tail(8)
                    fig_imp = px.bar(feat_imp, x='Peso', y='Fator', orientation='h',
                                     color_discrete_sequence=[BRAND_COLORS['primary']])
                    fig_imp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          margin=dict(t=0, l=0, r=0, b=0))
                    st.plotly_chart(fig_imp, use_container_width=True)

            with col_risk:
                st.markdown("##### 🚨 Risco por Departamento")
                df_risk = df_view[df_view['Risco'].isin(['CRÍTICO', 'ALERTA'])]
                rc = df_risk.groupby('Department').size().reset_index(name='Qtd')
                fig_r = px.bar(rc, x='Qtd', y='Department', orientation='h',
                               color_discrete_sequence=[BRAND_COLORS['danger']])
                fig_r.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    margin=dict(t=0, l=0, r=0, b=0))
                st.plotly_chart(fig_r, use_container_width=True)

            # 2. Distribuições Cruzadas
            st.markdown("---")
            g_dem1, g_dem2 = st.columns(2)

            with g_dem1:
                st.markdown("##### Perfil de Risco por Idade")
                fig_age_risk = px.histogram(df_view, x='Age', color='Risco', nbins=15,
                                            color_discrete_map={'CRÍTICO': BRAND_COLORS['danger'],
                                                                'ALERTA': BRAND_COLORS['warning'],
                                                                'BAIXO': BRAND_COLORS['accent']},
                                            barmode='stack')
                fig_age_risk.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', bargap=0.1)
                st.plotly_chart(fig_age_risk, use_container_width=True)

            with g_dem2:
                st.markdown("##### Salário vs Nível de Risco")
                fig_box_risk = px.box(df_view, x='Risco', y='MonthlyIncome', color='Risco',
                                      color_discrete_map={'CRÍTICO': BRAND_COLORS['danger'],
                                                          'ALERTA': BRAND_COLORS['warning'],
                                                          'BAIXO': BRAND_COLORS['accent']})
                fig_box_risk.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                           showlegend=False)
                st.plotly_chart(fig_box_risk, use_container_width=True)

            # 3. Tabela
            st.markdown("##### 📋 Colaboradores em Risco")
            st.dataframe(df_view[df_view['Probabilidade'] >= threshold_user][
                             ['EmployeeNumber', 'Department', 'JobRole', 'MonthlyIncome', 'Probabilidade',
                              'Risco']].sort_values('Probabilidade', ascending=False), use_container_width=True)

    # =========================================
    # ABA 3: SIMULADOR
    # =========================================
    with tab3:
        st.markdown("##### Simulador What-If")
        st.caption("Simule um perfil e descubra qual variável tem maior impacto no risco.")

        with st.form("sim_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age_in = st.number_input("Idade", 18, 70, 30)
                dept_in = st.selectbox("Departamento", ['Sales', 'R&D', 'HR'])
                role_in = st.selectbox("Cargo", ['Sales Executive', 'Research Scientist', 'Manager'])
            with c2:
                inc_in = st.number_input("Salário (R$)", 1500, 50000, 4500)
                ot_in = st.selectbox("Hora Extra?", ['Yes', 'No'])
                dist_in = st.slider("Distância", 1, 50, 10)
            with c3:
                env_in = st.slider("Ambiente (1-4)", 1, 4, 3)
                yrs_in = st.number_input("Anos de Empresa", 0, 40, 5)
                mar_in = st.selectbox("Estado Civil", ['Single', 'Married', 'Divorced'])

            if st.form_submit_button("🔍 CALCULAR RISCO & IMPACTO"):
                base_dict = {
                    'Age': [age_in], 'Department': [dept_in], 'JobRole': [role_in], 'MonthlyIncome': [inc_in],
                    'OverTime': [ot_in], 'DistanceFromHome': [dist_in], 'EnvironmentSatisfaction': [env_in],
                    'TotalWorkingYears': [yrs_in], 'YearsAtCompany': [yrs_in], 'MaritalStatus': [mar_in],
                    'BusinessTravel': ['Travel_Rarely'], 'Gender': ['Male'], 'EducationField': ['Life Sciences']
                }
                df_base = pd.DataFrame(base_dict)
                prob_base = process_and_predict(df_base, model, train_features)[0]

                # ANÁLISE DE SENSIBILIDADE
                scenarios = []
                # 1. Hora Extra
                new_ot = 'No' if ot_in == 'Yes' else 'Yes'
                df_ot = df_base.copy();
                df_ot['OverTime'] = new_ot
                prob_ot = process_and_predict(df_ot, model, train_features)[0]
                scenarios.append({'Mudança': f'Hora Extra -> {new_ot}', 'Impacto': prob_ot - prob_base})

                # 2. Salário
                df_inc = df_base.copy();
                df_inc['MonthlyIncome'] = inc_in * 1.2
                prob_inc = process_and_predict(df_inc, model, train_features)[0]
                scenarios.append({'Mudança': 'Salário +20%', 'Impacto': prob_inc - prob_base})

                # 3. Ambiente
                if env_in < 4:
                    df_env = df_base.copy();
                    df_env['EnvironmentSatisfaction'] = env_in + 1
                    prob_env = process_and_predict(df_env, model, train_features)[0]
                    scenarios.append({'Mudança': 'Ambiente +1 pt', 'Impacto': prob_env - prob_base})

                st.divider()
                cr1, cr2 = st.columns([1, 2])
                with cr1:
                    lbl = "ALTO" if prob_base > threshold_user else "BAIXO"
                    color_delta = "neg" if prob_base > threshold_user else "pos"
                    kpi_card("Probabilidade Atual", f"{prob_base:.1%}", lbl, color_delta)

                with cr2:
                    st.markdown("##### 📉 Impacto de Mudanças")
                    df_sens = pd.DataFrame(scenarios).sort_values('Impacto')
                    fig_torn = px.bar(df_sens, x='Impacto', y='Mudança', orientation='h',
                                      text_auto='.1%',
                                      color='Impacto',
                                      color_continuous_scale=[BRAND_COLORS['accent'], BRAND_COLORS['danger']])
                    fig_torn.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_torn, use_container_width=True)

    # =========================================
    # ABA 4: ROI
    # =========================================
    with tab4:
        st.markdown("##### Calculadora ROI")
        if 'resultado_ia' in st.session_state:
            df_roi = st.session_state['resultado_ia']
            c1, c2, c3 = st.columns(3)
            cr = c1.number_input("Custo Reposição", 40000)
            ci = c2.number_input("Custo Intervenção", 500)
            ef = c3.slider("Taxa Sucesso %", 0, 100, 50) / 100

            tgt = len(df_roi[df_roi['Probabilidade'] >= threshold_user])
            sav = int(tgt * 0.45 * ef)
            roi = (sav * cr) - (tgt * ci)

            fig_w = go.Figure(go.Waterfall(measure=["relative", "relative", "total"], x=["Economia", "Custo", "ROI"],
                                           y=[sav * cr, -tgt * ci, roi], connector={"line": {"color": "#cbd5e1"}},
                                           decreasing={"marker": {"color": BRAND_COLORS['danger']}},
                                           increasing={"marker": {"color": BRAND_COLORS['accent']}},
                                           totals={"marker": {"color": BRAND_COLORS['primary']}}))
            fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_w, use_container_width=True)
            kpi_card("ROI Líquido", f"R$ {roi / 1000:,.0f}k")
        else:
            st.info("Execute a Aba 2 primeiro.")