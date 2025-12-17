"""Página Inicial"""
import streamlit as st
from streamlit.components.v1 import html



# --- Google Analytics ---
GA_ID = "G-E922YWBZM7"  # substitua pelo seu ID real

GA_SCRIPT = f"""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-E922YWBZM7"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){{dataLayer.push(arguments);}}
  gtag('js', new Date());
  gtag('config', '{GA_ID}');
</script>
"""

st.set_page_config(
    page_title="Home", 
    layout="wide",initial_sidebar_state="collapsed")

#Corpo da página
#st.markdown(GA_SCRIPT, unsafe_allow_html=True)
html(GA_SCRIPT, height=0)

st.markdown("""
<style>
h1, h2, h3, h4, h5, h6 {
    font-weight: 600;
}
.main-text {
    background: rgba(255, 255, 255, 0.6);
    padding: 20px 30px;
    border-radius: 12px;
    border-left: 4px solid #4A90E2;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
ul li {
    line-height: 1.6;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="main-text">

## 📘 Calculadora de Métodos Numéricos Computacionais

Esta aplicação reúne um conjunto de algoritmos numéricos utilizados na disciplina  
**Métodos Numéricos Computacionais**, parte do currículo do curso de **Engenharia Elétrica**  
do **CEFET-MG – Campus Nepomuceno**.

O objetivo é oferecer um ambiente integrado para estudo, prática e verificação  
dos métodos apresentados em aula.  

Os conteúdos estão organizados nas seguintes categorias:

- **Sistemas Lineares**
- **Integração Numérica**
- **Interpolação**
- **Ajuste de Curvas**
- **Raízes de Equações**
- **Equações Diferenciais Ordinárias**

</div>

<br>

#### 👉 Selecione uma categoria no menu lateral e explore as funcionalidades disponíveis.
""", unsafe_allow_html=True)

#Rodapé
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        padding: 12px;
        background: rgba(240, 240, 240, 0.8);
        backdrop-filter: blur(6px);
        font-size: 14px;
        color: #444;
        text-align: center;
        border-top: 1px solid #ccc;
    }
    </style>

    <div class="footer">
        © 2025 • Calculadora MNC • Desenvolvida por <b>Rosana Massahud</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    iframe[title="Google Analytics"] {display:none;}
    </style>
    """,
    unsafe_allow_html=True
)
