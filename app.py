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
# 1. BRAND SYSTEM (ARQDIGITAL / RH SYSTEM PRO)
# ==========================================
BRAND_COLORS = {
    'primary': '#0E1F26',  # Azul Petróleo
    'accent': '#2FA4A9',  # Verde Técnico
    'light': '#E3E6E8',  # Cinza Claro
    'bg_light': '#F7F9FA',  # Fundo da Página
    'card_bg': '#FFFFFF',  # Fundo dos Cards
    'danger': '#D9534F',  # Vermelho
    'warning': '#F59E0B',  # Amarelo
    'neutral': '#8A8F95',  # Cinza Texto
    'text_white': '#FFFFFF'  # Texto Sidebar
}

# --- MOCKS ---
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
# 2. CSS ENGINE (CORREÇÃO DOS INPUTS E BOTÕES)
# ==========================================
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* TEMA GERAL */
    .stApp {{
        background-color: {BRAND_COLORS['bg_light']};
        color: {BRAND_COLORS['primary']};
        font-family: 'Inter', sans-serif;
    }}

    /* SIDEBAR */
    [data-testid="stSidebar"] {{
        background-color: {BRAND_COLORS['primary']};
    }}
    [data-testid="stSidebar"] * {{
        color: {BRAND_COLORS['text_white']} !important;
    }}

    /* --- CORREÇÃO 1: INPUTS NUMÉRICOS (REMOVER BLOCO PRETO) --- */
    /* Força fundo branco em toda a estrutura do input */
    div[data-testid="stNumberInput"] div[data-baseweb="input"] {{
        background-color: #FFFFFF !important;
        border: 1px solid {BRAND_COLORS['light']} !important;
    }}

    /* Input de texto real */
    div[data-testid="stNumberInput"] input {{
        color: {BRAND_COLORS['primary']} !important;
        background-color: transparent !important;
    }}

    /* Botões laterais (+/-) que estavam pretos */
    div[data-testid="stNumberInput"] button {{
        background-color: #FFFFFF !important;
        color: {BRAND_COLORS['primary']} !important;
        border-left: 1px solid {BRAND_COLORS['light']} !important;
    }}
    /* Ícones das setinhas */
    div[data-testid="stNumberInput"] button svg {{
        fill: {BRAND_COLORS['primary']} !important;
    }}

    /* --- OUTROS INPUTS (Selectbox, TextInput) --- */
    div[data-baseweb="select"] > div, 
    div[data-baseweb="base-input"] {{
        background-color: #FFFFFF !important;
        color: {BRAND_COLORS['primary']} !important;
        border-color: {BRAND_COLORS['light']} !important;
    }}
    div[data-baseweb="select"] span {{
        color: {BRAND_COLORS['primary']} !important;
    }}
    div[data-baseweb="select"] svg {{
        fill: {BRAND_COLORS['primary']} !important;
    }}

    /* --- CORREÇÃO 2: BOTÕES (SIDEBAR E PRINCIPAL) --- */
    /* Força cor de fundo Teal e Texto Branco com !important para sobrescrever temas */
    .stButton > button {{
        background-color: {BRAND_COLORS['accent']} !important;
        color: white !important;
        border: none;
        font-weight: 600;
        height: 3em;
        border-radius: 4px;
    }}
    .stButton > button:hover {{
        background-color: #248c91 !important;
        color: white !important;
    }}

    /* Títulos e Métricas */
    h1, h2, h3, h4, h5 {{ color: {BRAND_COLORS['primary']} !important; }}

    .metric-card {{
        background-color: {BRAND_COLORS['card_bg']};
        border-left: 5px solid {BRAND_COLORS['accent']};
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }}
    .metric-label {{
        font-size: 11px;
        font-weight: 700;
        color: {BRAND_COLORS['neutral']} !important;
        text-transform: uppercase;
    }}
    .metric-value {{
        font-size: 28px;
        font-weight: 800;
        color: {BRAND_COLORS['primary']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# --- HELPER ---
def kpi_card(label, value, delta=None, color="neutral"):
    c_delta = BRAND_COLORS['neutral']
    colors = {"pos": BRAND_COLORS['accent'], "neg": BRAND_COLORS['danger'], "warn": BRAND_COLORS['warning']}
    if color in colors: c_delta = colors[color]

    delta_html = f"<div style='font-size:12px; font-weight:700; color:{c_delta}; margin-top:5px;'>{delta}</div>" if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)


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
    except:
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
# LÓGICA PRINCIPAL
# ==========================================

with st.sidebar:
    st.markdown("### RH System Pro")
    st.caption("PEOPLE ANALYTICS ENTERPRISE")
    st.markdown(
        f"<div style='background:rgba(255,255,255,0.1); padding:10px; border-radius:5px; margin-bottom:20px;'><span style='color:{BRAND_COLORS['accent']}; font-weight:bold;'>🟢 SISTEMA ONLINE</span></div>",
        unsafe_allow_html=True)
    st.markdown("### ⚙️ Calibragem")
    threshold_user = st.slider("Sensibilidade", 0.0, 1.0, 0.40, 0.05)
    st.divider()
    if st.button("🔄 RECARREGAR SISTEMA"):
        st.session_state.clear()
        st.rerun()

st.title("Painel de Estratégia de Talentos")
st.markdown("Visão integrada de dados e decisões humanas.")
st.divider()

model, train_features = load_model_system()

if 'dados_rh' not in st.session_state or st.session_state['dados_rh'] is None:
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.info("ℹ️ Conecte-se ao Data Warehouse.")
        if st.button("CONECTAR BASE DE DADOS", use_container_width=True):
            try:
                path = "2026-01-19T00-31_export.csv"
                if not pathlib.Path(path).exists(): path = str(config.NEW_DATA_FILE)
                st.session_state['dados_rh'] = pd.read_csv(path)
                st.rerun()
            except Exception as e:
                st.error(f"Erro: {e}")
else:
    df_input = st.session_state['dados_rh']
    tab1, tab2, tab3, tab4 = st.tabs(["📊 VISÃO GERAL", "🔮 PREDIÇÃO & RISCO", "🧪 SIMULADOR", "💰 ROI"])

    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        with c1: kpi_card("Headcount", f"{len(df_input):,.0f}", "+120", "pos")
        with c2: kpi_card("Idade Média", f"{df_input['Age'].mean():.0f} Anos")
        with c3: kpi_card("Salário Médio", f"R$ {df_input['MonthlyIncome'].mean():,.0f}", "+4%", "neg")
        with c4: kpi_card("Tempo de Casa", f"{df_input['YearsAtCompany'].mean():.1f} Anos")

        g1, g2 = st.columns(2)
        with g1:
            st.markdown("##### Distribuição por Departamento")
            fig_d = px.histogram(df_input, x='Department', color='Department',
                                 color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['accent']])
            fig_d.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color=BRAND_COLORS['primary'])
            st.plotly_chart(fig_d, use_container_width=True)
        with g2:
            st.markdown("##### Gênero")
            fig_p = px.pie(df_input, names='Gender', hole=0.6,
                           color_discrete_sequence=[BRAND_COLORS['accent'], BRAND_COLORS['light']])
            fig_p.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color=BRAND_COLORS['primary'])
            st.plotly_chart(fig_p, use_container_width=True)

        g3, g4 = st.columns(2)
        with g3:
            st.markdown("##### Demografia (Idade)")
            fig_a = px.histogram(df_input, x='Age', nbins=20, color_discrete_sequence=[BRAND_COLORS['primary']])
            fig_a.update_layout(bargap=0.1, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color=BRAND_COLORS['primary'])
            st.plotly_chart(fig_a, use_container_width=True)
        with g4:
            st.markdown("##### Faixa Salarial")
            fig_b = px.box(df_input, x='Department', y='MonthlyIncome', color='Department',
                           color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['accent']])
            fig_b.update_layout(showlegend=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color=BRAND_COLORS['primary'])
            st.plotly_chart(fig_b, use_container_width=True)

        st.markdown("##### Correlação: Idade vs Renda")
        fig_s = px.scatter(df_input.sample(min(2000, len(df_input))), x='Age', y='MonthlyIncome', color='Department',
                           color_discrete_sequence=[BRAND_COLORS['primary'], BRAND_COLORS['accent'],
                                                    BRAND_COLORS['warning']])
        fig_s.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color=BRAND_COLORS['primary'])
        st.plotly_chart(fig_s, use_container_width=True)

    with tab2:
        c_l, c_r = st.columns([3, 1])
        with c_l:
            st.markdown("##### Diagnóstico de Risco (XGBoost)")
            st.caption(f"Threshold Atual: {threshold_user:.0%}")
        with c_r:
            if st.button("⚡ RODAR MODELO", use_container_width=True):
                probs = process_and_predict(df_input, model, train_features)
                df_res = df_input.copy()
                df_res['Probabilidade'] = probs
                df_res['Risco'] = df_res['Probabilidade'].apply(
                    lambda x: 'CRÍTICO' if x >= 0.7 else ('ALERTA' if x >= threshold_user else 'BAIXO'))
                st.session_state['resultado_ia'] = df_res

        if 'resultado_ia' in st.session_state:
            res = st.session_state['resultado_ia']
            k1, k2, k3 = st.columns(3)
            with k1:
                kpi_card("Crítico", len(res[res['Risco'] == 'CRÍTICO']), "Ação Imediata", "neg")
            with k2:
                kpi_card("Alerta", len(res[res['Risco'] == 'ALERTA']), "Monitorar", "warn")
            with k3:
                kpi_card("Estável", len(res[res['Risco'] == 'BAIXO']), "Seguro", "pos")
            st.divider()

            col_fi, col_rd = st.columns(2)
            with col_fi:
                st.markdown("##### 🔍 Top Fatores")
                if hasattr(model, 'feature_importances_'):
                    feat_imp = pd.DataFrame({'Fator': train_features, 'Peso': model.feature_importances_}).sort_values(
                        'Peso').tail(10)
                    fig_fi = px.bar(feat_imp, x='Peso', y='Fator', orientation='h',
                                    color_discrete_sequence=[BRAND_COLORS['primary']])
                    fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                         font_color=BRAND_COLORS['primary'], margin=dict(l=10, r=10, t=30, b=10))
                    st.plotly_chart(fig_fi, use_container_width=True)
            with col_rd:
                st.markdown("##### 🚨 Risco por Departamento")
                df_risk = res[res['Risco'].isin(['CRÍTICO', 'ALERTA'])]
                if not df_risk.empty:
                    rc = df_risk.groupby('Department').size().reset_index(name='Qtd')
                    fig_rc = px.bar(rc, x='Qtd', y='Department', orientation='h',
                                    color_discrete_sequence=[BRAND_COLORS['danger']])
                    fig_rc.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                         font_color=BRAND_COLORS['primary'])
                    st.plotly_chart(fig_rc, use_container_width=True)
                else:
                    st.info("Nenhum risco crítico identificado.")

            g_ar, g_sr = st.columns(2)
            with g_ar:
                st.markdown("##### Risco por Faixa Etária")
                # --- CORREÇÃO 3: GRÁFICO EMPILHADO (STACKED BAR) ---
                # Usar groupby + px.bar garante o empilhamento correto, resolvendo o problema visual
                df_age_risk = res.groupby(['Age', 'Risco']).size().reset_index(name='Qtd')
                fig_ar = px.bar(df_age_risk, x='Age', y='Qtd', color='Risco',
                                color_discrete_map={'CRÍTICO': BRAND_COLORS['danger'],
                                                    'ALERTA': BRAND_COLORS['warning'], 'BAIXO': BRAND_COLORS['accent']})
                fig_ar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font_color=BRAND_COLORS['primary'])
                st.plotly_chart(fig_ar, use_container_width=True)
            with g_sr:
                st.markdown("##### Salário vs Nível de Risco")
                fig_sr = px.box(res, x='Risco', y='MonthlyIncome', color='Risco',
                                color_discrete_map={'CRÍTICO': BRAND_COLORS['danger'],
                                                    'ALERTA': BRAND_COLORS['warning'], 'BAIXO': BRAND_COLORS['accent']})
                fig_sr.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font_color=BRAND_COLORS['primary'], showlegend=False)
                st.plotly_chart(fig_sr, use_container_width=True)

            st.markdown("##### 📋 Colaboradores em Risco")
            st.dataframe(res[res['Probabilidade'] >= threshold_user].sort_values('Probabilidade', ascending=False),
                         use_container_width=True)

    with tab3:
        st.markdown("#### Simulador de Cenários What-If")
        with st.form("sim_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                age_in = st.number_input("Idade", 18, 70, 30)
                dept_opts = sorted(df_input['Department'].unique().tolist())
                role_opts = sorted(df_input['JobRole'].unique().tolist())
                dept_in = st.selectbox("Departamento", dept_opts)
                role_in = st.selectbox("Cargo", role_opts)
            with c2:
                inc_in = st.number_input("Salário (R$)", 1500, 50000, 5000)
                ot_in = st.selectbox("Hora Extra?", ['Yes', 'No'])
                dist_in = st.slider("Distância", 1, 50, 10)
            with c3:
                env_in = st.slider("Ambiente (1-4)", 1, 4, 3)
                yrs_in = st.number_input("Anos de Empresa", 0, 40, 5)
                mar_in = st.selectbox("Estado Civil", sorted(df_input['MaritalStatus'].unique().tolist()))

            if st.form_submit_button("🔍 CALCULAR RISCO & IMPACTO"):
                base_d = {'Age': [age_in], 'Department': [dept_in], 'JobRole': [role_in], 'MonthlyIncome': [inc_in],
                          'OverTime': [ot_in], 'DistanceFromHome': [dist_in], 'EnvironmentSatisfaction': [env_in],
                          'TotalWorkingYears': [yrs_in], 'YearsAtCompany': [yrs_in], 'MaritalStatus': [mar_in],
                          'BusinessTravel': ['Travel_Rarely'], 'Gender': ['Male'], 'EducationField': ['Life Sciences']}
                df_b = pd.DataFrame(base_d)
                prob_b = process_and_predict(df_b, model, train_features)[0]

                scen = []
                new_ot = 'No' if ot_in == 'Yes' else 'Yes'
                df_ot = df_b.copy();
                df_ot['OverTime'] = new_ot
                scen.append({'Mudança': f'Hora Extra -> {new_ot}',
                             'Impacto': process_and_predict(df_ot, model, train_features)[0] - prob_b})

                df_inc = df_b.copy();
                df_inc['MonthlyIncome'] = inc_in * 1.2
                scen.append({'Mudança': 'Salário +20%',
                             'Impacto': process_and_predict(df_inc, model, train_features)[0] - prob_b})

                st.divider()
                cr1, cr2 = st.columns([1, 2])
                with cr1:
                    lbl = "ALTO" if prob_b > threshold_user else "BAIXO"
                    color_delta = "neg" if prob_b > threshold_user else "pos"
                    kpi_card("Risco Atual", f"{prob_b:.1%}", lbl, color_delta)
                with cr2:
                    df_s = pd.DataFrame(scen).sort_values('Impacto')
                    fig_t = px.bar(df_s, x='Impacto', y='Mudança', orientation='h', text_auto='.1%', color='Impacto',
                                   color_continuous_scale=[BRAND_COLORS['accent'], BRAND_COLORS['danger']])
                    fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        font_color=BRAND_COLORS['primary'], uniformtext_minsize=8,
                                        uniformtext_mode='hide')
                    fig_t.update_traces(textfont_color='white')
                    st.plotly_chart(fig_t, use_container_width=True)

    with tab4:
        st.markdown("#### Calculadora ROI")
        if 'resultado_ia' in st.session_state:
            df_roi = st.session_state['resultado_ia']
            c_r1, c_r2, c_r3 = st.columns(3)
            c_rep = c_r1.number_input("Custo Reposição", 40000)
            c_int = c_r2.number_input("Custo Intervenção", 500)
            ef_rt = c_r3.slider("Taxa Sucesso %", 0, 100, 50) / 100

            tgt = len(df_roi[df_roi['Probabilidade'] >= threshold_user])
            sav = int(tgt * 0.45 * ef_rt)
            roi_l = (sav * c_rep) - (tgt * c_int)

            fig_w = go.Figure(go.Waterfall(measure=["relative", "relative", "total"], x=["Economia", "Custo", "ROI"],
                                           y=[sav * c_rep, -tgt * c_int, roi_l],
                                           connector={"line": {"color": "#cbd5e1"}},
                                           decreasing={"marker": {"color": BRAND_COLORS['danger']}},
                                           increasing={"marker": {"color": BRAND_COLORS['accent']}},
                                           totals={"marker": {"color": BRAND_COLORS['primary']}}))
            fig_w.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font_color=BRAND_COLORS['primary'])
            st.plotly_chart(fig_w, use_container_width=True)
            kpi_card("ROI Líquido", f"R$ {roi_l / 1000:,.0f}k")