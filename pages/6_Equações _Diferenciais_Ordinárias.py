import streamlit as st
from methods.edos import euler, runge_kutta_2, runge_kutta_3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="EDO's", page_icon="📈", layout='centered')
st.title("Resolução Numérica de Equações Diferenciais Ordinárias")

def plotar_grafico(df, titulo):
    fig = plt.figure()
    # Plotar o gráfico (opcional)
    plt.plot(df['x'], df['y'], marker='o')
    plt.title(titulo)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    st.plotly_chart(fig, use_container_width=True)

metodo = st.selectbox(
    "Escolha o método de ajuste de curvas",
    ("Método de Euler", "Método de Runge Kutta de 2ª ordem", "Método de Runge Kutta de 3ª ordem")
)

st.write("Método escolhido:", metodo)

str_fxy = st.text_input('f(x,y):', '(x**2-y**2)/(x*y)')
str_a = st.text_input('Limite inferior:', '1')
str_b = st.text_input('Limite superior:', '2')
str_n_subinter = st.number_input('Número de subintervalos:', value=10, min_value=2)
str_x0 = st.text_input('Valor inicial ($X_0$):', '1')
str_y0 = st.text_input('Valor inicial ($y_0$):', '2')

bt_aproximar = st.button('Aproximar')

if(bt_aproximar):
    fxy = lambda x,y:eval(str_fxy)
    a = int(eval(str_a))
    b = int(eval(str_b))
    n = int(str_n_subinter)
    x0 = float(eval(str_x0))
    y0 = float(eval(str_y0))

    if(metodo=='Método de Euler'):
        try:
            h, tabela = euler(fxy,a,b,n,x0,y0)
            st.subheader('Aproximação pelo Método de Euler')
            st.write(f'Tamanho do passo: {h:.4}')
            st.dataframe(tabela)
            plotar_grafico(tabela,'Aproximação pelo Método de Euler')
            
        except Exception as e:
            st.error(f"Erro: {e}")

    elif(metodo=='Método de Runge Kutta de 2ª ordem'):
        try:
            h, tabela = runge_kutta_2(fxy,x0, y0, a,b,n)
            st.subheader('Aproximação pelo Método de Runge Kutta de 2ª ordem')
            st.subheader('Euler Perfeiçoado')
            st.write(f'Tamanho do passo: {h:.4}')
            st.dataframe(tabela)
            plotar_grafico(tabela,'Método de Runge-Kutta de 2ª Ordem')
        except Exception as e:
            st.error(f"Erro: {e}")
    elif(metodo=='Método de Runge Kutta de 3ª ordem'):
        try:
            h, tabela = runge_kutta_3(fxy,x0, y0, a,b,n)
            st.subheader('Aproximação pelo Método de Runge Kutta de 3ª ordem')
            st.write(f'Tamanho do passo: {h:.4}')
            st.dataframe(tabela)
            plotar_grafico(tabela,'Método de Runge-Kutta de 3ª Ordem')
        except Exception as e:
            st.error(f"Erro: {e}")