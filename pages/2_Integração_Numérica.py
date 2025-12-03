"""Integração Numérica"""

import streamlit as st
from methods.integracao import trapezio_simples, trapezio_repetido, simpson_1, simpson_38, quadratura_gauss
import numpy as np
import pandas as pd
from scipy.integrate import quad

st.set_page_config(page_title='Integração Numérica', page_icon="📈", layout='centered')
metodo = st.selectbox(
    "Escolha o método de integração numérica",
    ("Trapézio simples", "Trapézio c/ repetição", "1/3 de Simpson",
     "3/8 de Simpson", "Quadratura Gaussiana")
)

if(metodo=='Quadratura Gaussiana'):
    func_str = st.text_input("f(x):", "2*x**3+3*x**2+6*x+1")
    a = st.number_input("a:", value=1)
    b = st.number_input("b:", value=5)
    n = st.number_input("Número de pontos:", value=2, min_value=2) 
else:
    func_str = st.text_input("f(x):", "1/x")
    a = st.number_input("a:", value=1)
    b = st.number_input("b:", value=3)
    if(metodo=='Trapézio simples'):
        n = st.number_input("Número de subintervalos:", value=1, min_value=1, max_value=1)
    elif(metodo=='Trapézio c/ repetição' or metodo == '1/3 de Simpson'):
        n = st.number_input("Número de subintervalos:", value=2, min_value=2)
    elif(metodo=='3/8 de Simpson'):
        n = st.number_input("Número de subintervalos:", value=3, min_value=3)

n = int(n)
h = float((b-a)/n)

f = lambda x:eval(func_str)
x = np.linspace(a, b, n+1, dtype=float)
y = [float(f(xi)) for xi in x]

valor_real, erro_vreal = quad(f, a, b)
erro_vreal = np.rint(erro_vreal)

df = pd.DataFrame(
    {
        "x": x,
        "y": y,
    }
)

st.line_chart(df, x="x", y="y")

integrar = st.button("Integrar")

if (integrar):
    resultado = 0

    try:
        if(metodo == 'Trapézio simples'):
            resultado = trapezio_simples(x,y)
        elif(metodo == 'Trapézio c/ repetição'):
            resultado = trapezio_repetido(y,h)
        elif(metodo == '1/3 de Simpson'):
            resultado = simpson_1(y,n+1,h)
        elif(metodo == '3/8 de Simpson'):
            resultado = simpson_38(y,n+1,h)
        elif(metodo == 'Quadratura Gaussiana'):
            resultado, erro, tabela = quadratura_gauss(f,a,b,n,True)
        if(metodo=='Quadratura Gaussiana'):
            st.subheader('Tabela resultante do cálculo dos pesos', divider='green')
            st.dataframe(tabela)
        result_str = f'Resultado ≈ {resultado:.6f}'
        st.success(result_str)
        if(erro_vreal==0):
            erro_absoluto = round(abs(valor_real-resultado),6)
            erro_relativo = round((erro_absoluto/valor_real),6)        
            st.markdown(
                f"""
                **Erro absoluto:** {erro_absoluto}  
                **Erro relativo:** {erro_relativo*100}%
                """
            )
            
        else:
            st.warning('Não foi possível calcular o resultado analítico da integral definida')
    except Exception as e:
        st.error(f"Erro: {e}")