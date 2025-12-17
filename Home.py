"""Página Inicial"""
import streamlit as st

from pathlib import Path

# =========================
# Contador simples de acessos
# =========================
COUNTER_FILE = Path("counter.txt")

if not COUNTER_FILE.exists():
    COUNTER_FILE.write_text("0")

if "counted" not in st.session_state:
    count = int(COUNTER_FILE.read_text())
    count += 1
    COUNTER_FILE.write_text(str(count))
    st.session_state["counted"] = True
else:
    count = int(COUNTER_FILE.read_text())

st.set_page_config(
    page_title="Calculadora MNC",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Estilos
# =========================
st.markdown(
    """
    <style>
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600;
    }

    .main-text {
        background: rgba(255, 255, 255, 0.6);
        padding: 24px 32px;
        border-radius: 12px;
        border-left: 5px solid #4A90E2;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        max-width: 900px;
        margin: auto;
    }

    ul li {
        line-height: 1.8;
        margin-bottom: 6px;
    }

    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        padding: 12px;
        background: rgba(240, 240, 240, 0.85);
        backdrop-filter: blur(6px);
        font-size: 14px;
        color: #444;
        text-align: center;
        border-top: 1px solid #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Conteúdo principal
# =========================
st.markdown(
    """
    <div class="main-text">

    ## 📘 Calculadora de Métodos Numéricos Computacionais

    Esta aplicação reúne um conjunto de algoritmos numéricos utilizados na disciplina  
    <b>Métodos Numéricos Computacionais</b>, parte do currículo do curso de  
    <b>Engenharia Elétrica</b> do <b>CEFET-MG – Campus Nepomuceno</b>.

    O objetivo é oferecer um ambiente integrado para estudo, prática e verificação  
    dos métodos apresentados em aula.

    <br>

    <b>Os conteúdos estão organizados nas seguintes categorias:</b>

    <ul>
        <li><b>Sistemas Lineares</b></li>
        <li><b>Integração Numérica</b></li>
        <li><b>Interpolação</b></li>
        <li><b>Ajuste de Curvas</b></li>
        <li><b>Raízes de Equações</b></li>
        <li><b>Equações Diferenciais Ordinárias</b></li>
    </ul>

    </div>

    <br>

    <h4 style="text-align:center;">
    👉 Selecione uma categoria no menu lateral e explore as funcionalidades disponíveis.
    </h4>
    """,
    unsafe_allow_html=True
)

# =========================
# Rodapé
# =========================
st.markdown(
    f"""
    <div class="footer">
        © 2025 • Calculadora MNC • Desenvolvida por <b>Rosana Massahud</b>
        <br>
        Acessos: <b>{count}</b>
    </div>
    """,
    unsafe_allow_html=True
)