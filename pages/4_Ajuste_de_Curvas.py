import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from methods.ajuste_curvas import ajuste_linear, g_linear, ajuste_logaritmico, g_log, ajuste_exponencial, g_exponencial, ajuste_potencia, g_potencia, regressao_linear_EN

def plota_grafico(df_linha, df_pontos, titulo):
    fig = go.Figure()

    # Linha (df_linha)
    fig.add_trace(
        go.Scatter(
            x=df_linha["x"],
            y=df_linha["y"],
            mode="lines",
            name="Ajuste",
            line=dict(width=3)
        )
    )

    # Pontos (df_pontos)
    fig.add_trace(
        go.Scatter(
            x=df_pontos["x"],
            y=df_pontos["y"],
            mode="markers",
            name="Pontos Originais",
            marker=dict(size=8, symbol="circle")
        )
    )

    fig.update_layout(
        title=titulo,
        xaxis_title="x",
        yaxis_title="y",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

st.set_page_config(page_title='Ajuste de Curvas', page_icon="📈", layout='centered')
st.title("Ajuste de Curvas")

metodo = st.selectbox(
    "Escolha o método de ajuste de curvas",
    ("Regressão Linear", "Ajuste Logarítmico", "Ajuste Exponencial", "Ajuste Potência", "Regressão Linear Múltipla")
)

st.write("Método escolhido:", metodo)

if(metodo=='Regressão Linear' or metodo=='Ajuste Exponencial' or metodo=='Ajuste Logarítmico' or metodo=='Ajuste Potência'):
    str_exemplo_x = "[1,2,3,4]"
    str_exemplo_y = "[3,5,6,8]"

    str_x = st.text_input("Informe os pontos x:", str_exemplo_x)
    str_y = st.text_input("Informe os pontos y:", str_exemplo_y)

    bt_ajuste = st.button('Ajustar curva')

    if(bt_ajuste):
        if(metodo=='Regressão Linear'):
            x = eval(str_x)
            y = eval(str_y)
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            ab = ajuste_linear(x,y)
            st.success(rf'$y = {ab[0]:.4f}\cdot x +{ab[1]:.4f}$')
            st.markdown(
                    f"""
                    **R²:** {ab[2]:.4f}  
                    **a=** {ab[0]:.4f}  
                    **b=** {ab[1]:.4f}  
                    **$y = {ab[0]:.4f}\cdot x +{ab[1]:.4f}$**  
                    """
                )
            df_pontos = pd.DataFrame({"x": x,"y": y,})
            
            y_ = [g_linear(ab,xi) for xi in x]
            df_linha = pd.DataFrame({"x" : x,"y" : y_ })
            
            titulo = 'Ajuste Linear com Pontos Originais'

            plota_grafico(df_linha, df_pontos, titulo)

        elif(metodo=='Ajuste Logarítmico'):
            x = eval(str_x)
            y = eval(str_y)
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            ab = ajuste_logaritmico(x,y)
            
            st.success(rf'$y = {ab[0]:.4f}\cdot ln(x) +{ab[1]:.4f}$')
            st.markdown(
                    f"""
                    **R²:** {ab[2]:.4f}  
                    **a=** {ab[0]:.4f}  
                    **b=** {ab[1]:.4f}  
                    **$y = {ab[0]:.4f}\cdot ln(x) +{ab[1]:.4f}$**  
                    """
                )
            df_pontos = pd.DataFrame({"x": x,"y": y,})
            
            y_ = [g_log(ab,xi) for xi in x]
            df_linha = pd.DataFrame({"x" : x,"y" : y_ })
            
            titulo = 'Ajuste Logarítmico com Pontos Originais'

            plota_grafico(df_linha, df_pontos, titulo)

        elif(metodo=='Ajuste Exponencial'):
            x = eval(str_x)
            y = eval(str_y)
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            ab = ajuste_exponencial(x,y)
            b = ab[1]
            a = ab[0]

            st.success(
                rf"$y = {b:.4f}\cdot e^{{{a:.4f}x}}$"
            )
            st.markdown(
                    f"""
                    **R²:** {ab[2]:.4f}  
                    **a=** {ab[0]:.4f}  
                    **b=** {ab[1]:.4f}  
                    **$y = {b:.4f}\cdot e^{{{a:.4f}x}}$**    
                    """
                )
            df_pontos = pd.DataFrame({"x": x,"y": y,})
            
            y_ = [g_exponencial(ab,xi) for xi in x]
            df_linha = pd.DataFrame({"x" : x,"y" : y_ })
            
            titulo = 'Ajuste Exponencial com Pontos Originais'

            plota_grafico(df_linha, df_pontos, titulo)

        elif(metodo=='Ajuste Potência'):
            x = eval(str_x)
            y = eval(str_y)
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)

            ab = ajuste_potencia(x,y)
            b = ab[1]
            a = ab[0]

            st.success(
                rf"$y = {b:.4f}\cdot x^{{{a:.4f}}}$"
            )
            st.markdown(
                    f"""
                    **R²:** {ab[2]:.4f}  
                    **a=** {ab[0]:.4f}  
                    **b=** {ab[1]:.4f}  
                    **$y = {b:.4f}\cdot x^{{{a:.4f}}}$**    
                    """
                )
            df_pontos = pd.DataFrame({"x": x,"y": y,})
            
            y_ = [g_potencia(ab,xi) for xi in x]
            df_linha = pd.DataFrame({"x" : x,"y" : y_ })
            
            titulo = 'Ajuste Potência com Pontos Originais'

            plota_grafico(df_linha, df_pontos, titulo)
    
elif(metodo=='Regressão Linear Múltipla'):
    
    str_exemplo_x = "[[60.3, 61.1,60.2,61.2,63.2,63.6,65.0,63.8,66.0,67.9,68.2,66.5,68.7,69.6,69.3,70.6],[108.0,109.0,110.0,112.0,112.0,113.0,115.0,116.0,117.0,119.0,120.0,122.0,123.0,125.0,128.0,130.0]]"
    str_exemplo_y = "[234,259,258,285,329,347,365,363,396,419,443,445,483,503,518,555]"

    str_numero_pontos = st.number_input('Número de pontos: ', value=16)
    str_numero_var = st.number_input('Número de variáveis: ', value=2, min_value=2)
    str_numero_param = st.number_input('Número de parâmetros: ', value=3, min_value=1)
    str_x = st.text_area("Variáveis explicativas (x):", str_exemplo_x)
    str_y = st.text_area("Variáveis de resposta (y):", str_exemplo_y)

    bt_calc = st.button('Calcular parâmetros')
    if(bt_calc):
        n = int(str_numero_pontos)
        v = int(str_numero_var)
        p = int(str_numero_param)
        x = np.array(eval(str_exemplo_x), dtype=float)
        y = np.array(eval(str_exemplo_y), dtype=float)
        x = np.transpose(x)
        b, r2, sigma2, condErro = regressao_linear_EN(n,v,p,x,y)
        print('coef. de regressao: {}\ncoef. de determinacao: {}\nvariancia residual:{}\nErro:{}'.format(b, r2, sigma2, condErro))
        if(condErro):
            st.error('Erro ao calcular os parâmetros da regressão!')
        else:
            b_str = ", ".join(f"{coef:.6f}" for coef in b)

            st.markdown(f"""
            ### 📊 Resultados da Regressão
            **Coef. de regressão:** {b_str}  
            **Coef. de determinação (R²):** {r2:.6f}  
            **Variância residual:** {sigma2:.6f}  
            **Erro:** {condErro}
            """)



elif(metodo=='Regressão Polinomial'):
    pass