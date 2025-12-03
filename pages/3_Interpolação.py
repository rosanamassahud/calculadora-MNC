import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from methods.interpolacao import lagrange, metodo_newton, metodo_gregory_newton


st.set_page_config(page_title='Interpolação', page_icon="📈", layout='centered')
st.title("Interpolação")

metodo = st.selectbox(
    "Escolha o método de interpolação",
    ("Forma de Lagrange", "Polinômios de Newton", "Polinômios de Gregory-Newton")
)

st.write("Método escolhido:", metodo)

if(metodo == 'Polinômios de Gregory-Newton'):
    str_exemplo_x = '[-1,0,1]'
    str_exemplo_fx = '[4,1,-0.667]'
    str_exemplo_xi = '0.6'
else:
    str_exemplo_x = '[-1,0,2]'
    str_exemplo_fx = '[4,1,-1]'
    str_exemplo_xi = '1'

str_x = st.text_input('x:', str_exemplo_x)
str_fx = st.text_input('f(x):', str_exemplo_fx)
str_xi = st.text_input('Valor procurado (xi):', str_exemplo_xi)


x = list(eval(str_x))
y = list(eval(str_fx))
xi = float(eval(str_xi))

df = pd.DataFrame(
    {
        "x": x,
        "y": y,
    }
)

st.line_chart(df, x="x", y="y")

bt_interpolar = st.button('Interpolar')

if(bt_interpolar):
    yi = ' '
    
    try:

        if(xi>x[0] and xi<x[len(x)-1]):
            pass
        else:
            erro = 'Valor a ser interpolado não está dentro do intervalo de pontos'
            st.error(erro)
            raise(1, erro)

        if(metodo == 'Forma de Lagrange'):
            yi, erro = lagrange(x,y,xi)
        elif(metodo == 'Polinômios de Newton'):
            yi, tabela = metodo_newton(x,y,xi,True)
        elif(metodo == 'Polinômios de Gregory-Newton'):
            yi, tabela = metodo_gregory_newton(x,y,xi,True)

        if(yi == ' '):
            st.error('Não foi possível fazer a interpolação.')
        else:
            st.success('Valor interpolado: {}'.format(round(yi, 3)))
            if(metodo in ('Polinômios de Gregory-Newton', 'Polinômios de Newton')):
                st.subheader('Tabela de operadores de diferenças', divider='green')
                st.dataframe(tabela)

            # adicionar ponto
            df.loc[len(df)] = [xi, yi]

            # reordenar pelo eixo x
            df = df.sort_values(by="x").reset_index(drop=True)

            # gráfico base (linha + pontos)
            fig = px.line(df, x="x", y="y", markers=True)

            # adicionar o ponto interpolado, com estilo destacado
            fig.add_scatter(
                x=[xi],
                y=[yi],
                mode="markers",
                marker=dict(size=12, color="red", symbol="diamond"),
                name="Ponto interpolado"
            )

            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro: {e}")
