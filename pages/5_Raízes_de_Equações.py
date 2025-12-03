from methods.raizes import horner, metodo_bissecao, metodo_falsa_posicao, secante, newton_raphson, limites, ponto_fixo
import streamlit as st
import sympy as sp
import pandas as pd
import numpy as np
import plotly.express as px

def extrair_coeficientes(func_str):
    x = sp.symbols("x")
    expr = sp.sympify(func_str)
    poly = sp.Poly(expr, x)
    return poly.degree(), poly.all_coeffs()


st.set_page_config(page_title='Sistemas Lineares', page_icon="📈", layout='centered')
metodo = st.selectbox(
    "Escolha o método para determinar raízes de equações algébricas ou transcendentes",
    ("Avaliar polinômio (Horner)", "Determinar limites das raízes reais", "Método da Bisseção", "Método da Falsa Posição", "Método do Ponto Fixo",
     "Método de Newton-Raphson", "Método da Secante"),
)

st.write("Método escolhido:", metodo)

if(metodo == 'Avaliar polinômio (Horner)'):
    func_str = st.text_input("f(x):", "2*x**3+3*x**2+6*x+1")
    a_str = st.text_input("a:", "1")

    bt_avaliar = st.button('Avaliar')
    if(bt_avaliar):
        a = float(eval(a_str))

        # cria variável simbólica
        x = sp.symbols('x')

        # transforma string em expressão simbólica
        expr = sp.sympify(func_str)

        # grau do polinômio
        n = sp.degree(expr)

        # lista de coeficientes (do maior expoente pro menor)
        coeffs = sp.Poly(expr, x).all_coeffs()

        st.markdown(
                f"""
                **Grau:** {n}  
                **Coeficientes:** {coeffs}
                """
            )
        
        try:
            f_a = horner(n,coeffs,a)
            result_str = f'f(a) = {f_a:.2f}'
            st.success(result_str)
            min_lim = a - 5
            max_lim = a + 5
            
            f = lambda x:eval(func_str)
            x = np.linspace(int(min_lim), int(max_lim), 11, dtype=float)
            y = [float(f(xi)) for xi in x]
            
            df = pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                }
            )
            st.line_chart(df, x="x", y="y")
        except Exception as e:
            st.error(f"Erro: {e}")

elif(metodo=='Determinar limites das raízes reais'):
    func_str = st.text_input("f(x):", "x**3-3*x**2-6*x+8")
    n, coeffs = extrair_coeficientes(func_str)
    
    bt_calcular_lim = st.button('Calcular limites')

    if(bt_calcular_lim):
        st.markdown(
            f"""
            **Grau:** {n}  
            **Coeficientes:** {coeffs}
            """
        )
        try:
            L, msg = limites(n, coeffs)
            if(msg==""):
                limit_matrix = pd.DataFrame(
                    {
                        "Limite Inferior": [L[2], L[1]],
                        "Limite Superior": [L[3], L[0]],
                    },
                    index=["Raízes Negativas", "Raízes Positivas"],
                )
            else:
                limit_matrix = pd.DataFrame(
                    {
                        "Limite Inferior": [L[2], '-'],
                        "Limite Superior": [L[3], '-'],
                    },
                    index=["Raízes Negativas", "Raízes Positivas"],
                )
                st.write(msg)
            st.table(limit_matrix)

            #plotar um gráfico
            f = lambda x:eval(func_str)
            if(msg==""):
                n_ = int(L[0]-L[2])
                x = np.linspace(L[2], L[0], 2*(n_+1), dtype=float)
            else:
                n_ = int(1-L[2])
                x = np.linspace(L[2], 1, 2*(n_+1), dtype=float)
            y = [float(f(xi)) for xi in x]

            df = pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                }
            )

            st.line_chart(df, x="x", y="y")
        except Exception as e:
            st.error(f"Erro: {e}")

elif(metodo=='Método da Bisseção'):
    func_str = st.text_input("f(x):", "x**3-9*x+5")
    n, coeffs = extrair_coeficientes(func_str)
    a_str = st.text_input('a: ', '0.5')
    b_str = st.text_input('b: ', '1')
    toler_str = st.text_input('Tolerância: ', '0.01')
    max_iter_str = st.text_input('Máximo de iterações: ', '100')

    bt_calcular_lim = st.button('Calcular raiz')

    if(bt_calcular_lim):
        f = lambda x:eval(func_str)
        a = float(a_str)
        b = float(b_str)
        toler = float(toler_str)
        max_iter = int(max_iter_str)

        if(b>a):
            try:
                raiz, iter, tabela, msg_erro, erro  = metodo_bissecao(f, a, b, toler, max_iter)
                if(erro):
                    st.error(msg_erro)
                else:
                    st.success(f'Raiz: {raiz:.6f}')
                    st.write(f'Interações: {iter}')
                    st.dataframe(tabela)

                    #Plotar um gráfico com a raiz em destaque
                    x=np.arange(a-5, b+5)
                    y = [float(f(xi)) for xi in x]

                    df = pd.DataFrame(
                        {
                            "x": x,
                            "y": y,
                        }
                    )
                    # gráfico base (linha + pontos)
                    fig = px.line(df, x="x", y="y", markers=False)

                    # adicionar o ponto interpolado, com estilo destacado
                    fig.add_scatter(
                        x=[raiz],
                        y=[0],
                        mode="markers",
                        marker=dict(size=12, color="green", symbol="circle"),
                        name="Raiz"
                    )

                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro: {e}")

        else:
            st.warning('Verifique o intervalo para o cálculo da raiz.')

elif(metodo=='Método da Falsa Posição'):
    func_str = st.text_input("f(x):", "x**3-9*x+5")
    n, coeffs = extrair_coeficientes(func_str)
    a_str = st.text_input('a: ', '0.5')
    b_str = st.text_input('b: ', '1')
    toler_str = st.text_input('Tolerância: ', '0.01')
    max_iter_str = st.text_input('Máximo de iterações: ', '100')

    bt_calcular_lim = st.button('Calcular raiz')

    if(bt_calcular_lim):
        f = lambda x:eval(func_str)
        a = float(a_str)
        b = float(b_str)
        toler = float(toler_str)
        max_iter = int(max_iter_str)

        if(b>a):
            try:
                raiz, iter, tabela, msg_erro, erro  = metodo_falsa_posicao(f, a, b, toler, max_iter)
                if(erro):
                    st.error(msg_erro)
                else:
                    st.success(f'Raiz: {raiz:.6f}')
                    st.write(f'Interações: {iter}')
                    st.dataframe(tabela)

                    #Plotar um gráfico com a raiz em destaque
                    x=np.arange(a-5, b+5)
                    y = [float(f(xi)) for xi in x]

                    df = pd.DataFrame(
                        {
                            "x": x,
                            "y": y,
                        }
                    )
                    # gráfico base (linha + pontos)
                    fig = px.line(df, x="x", y="y", markers=False)

                    # adicionar o ponto interpolado, com estilo destacado
                    fig.add_scatter(
                        x=[raiz],
                        y=[0],
                        mode="markers",
                        marker=dict(size=12, color="green", symbol="circle"),
                        name="Raiz"
                    )

                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro: {e}")

        else:
            st.warning('Verifique o intervalo para o cálculo da raiz.')

elif(metodo=='Método do Ponto Fixo'):
    f_str = st.text_input('f(x):', 'x**3-9*x+5')
    g_str = st.text_input("g(x), tal que x = g(x):", "(9*x - 5)**(1/3)")
    #n, coeffs = extrair_coeficientes(func_str)
    a_str = st.text_input('a: ', '2.5')
    b_str = st.text_input('b: ', '3')
    
    toler_str = st.text_input('Tolerância: ', '0.01')
    max_iter_str = st.text_input('Máximo de iterações: ', '100')

    bt_calcular_lim = st.button('Calcular raiz')

    if(bt_calcular_lim):
        f = lambda x:eval(f_str)
        g = lambda x:eval(g_str)
        a = float(a_str)
        b = float(b_str)
        
        toler = float(toler_str)
        max_iter = int(max_iter_str)

        try:
            raiz, iter, tabela, msg_erro, erro  = ponto_fixo(f, g, a, b, toler, max_iter)
            if(erro):
                st.error(msg_erro)
                st.markdown("""
<div style="
    background-color:#fff3cd;
    color:#856404;
    padding:15px;
    border-radius:10px;
    border:1px solid #ffeeba;
    font-size:18px;">
<strong>⚠️ Método do Ponto Fixo</strong><br><br>
O método do ponto fixo converge apenas se:<br>
<span style="font-size:22px;"><strong>g'(x) &lt; 1</strong></span><br><br>
Tente escolher outra forma para <strong>g(x)</strong>.
</div>
""", unsafe_allow_html=True)
            else:
                st.success(f'Raiz: {raiz:.6f}')
                st.write(f'Interações: {iter}')
                st.dataframe(tabela)

                #Plotar um gráfico com a raiz em destaque
                x=np.arange(a-5, raiz+5)
                y = [float(f(xi)) for xi in x]

                df = pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                    }
                )
                # gráfico base (linha + pontos)
                fig = px.line(df, x="x", y="y", markers=False)

                # adicionar o ponto interpolado, com estilo destacado
                fig.add_scatter(
                    x=[raiz],
                    y=[f(raiz)],
                    mode="markers",
                    marker=dict(size=12, color="green", symbol="circle"),
                    name="Raiz"
                )

                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro: {e}")

elif(metodo=='Método de Newton-Raphson'):
    func_str = st.text_input("f(x):", "x**3-9*x+5")
    n, coeffs = extrair_coeficientes(func_str)
    a_str = st.text_input('x_0: ', '0.75')
    
    toler_str = st.text_input('Tolerância: ', '0.01')
    max_iter_str = st.text_input('Máximo de iterações: ', '100')

    bt_calcular_lim = st.button('Calcular raiz')

    if(bt_calcular_lim):
        f = lambda x:eval(func_str)
        a = float(a_str)
        
        toler = float(toler_str)
        max_iter = int(max_iter_str)

        try:
            raiz, iter, tabela, msg_erro, erro  = newton_raphson(f, a, toler, max_iter)
            if(erro):
                st.error(msg_erro)
            else:
                st.success(f'Raiz: {raiz:.6f}')
                st.write(f'Interações: {iter}')
                st.dataframe(tabela)

                #Plotar um gráfico com a raiz em destaque
                x=np.arange(a-5, raiz+5)
                y = [float(f(xi)) for xi in x]

                df = pd.DataFrame(
                    {
                        "x": x,
                        "y": y,
                    }
                )
                # gráfico base (linha + pontos)
                fig = px.line(df, x="x", y="y", markers=False)

                # adicionar o ponto interpolado, com estilo destacado
                fig.add_scatter(
                    x=[raiz],
                    y=[0],
                    mode="markers",
                    marker=dict(size=12, color="green", symbol="circle"),
                    name="Raiz"
                )

                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro: {e}")

elif(metodo=='Método da Secante'):
    func_str = st.text_input("f(x):", "x**3-9*x+5")
    n, coeffs = extrair_coeficientes(func_str)
    a_str = st.text_input('a: ', '0.5')
    b_str = st.text_input('b: ', '1')
    toler_str = st.text_input('Tolerância: ', '0.01')
    max_iter_str = st.text_input('Máximo de iterações: ', '100')

    bt_calcular_lim = st.button('Calcular raiz')

    if(bt_calcular_lim):
        f = lambda x:eval(func_str)
        a = float(a_str)
        b = float(b_str)
        toler = float(toler_str)
        max_iter = int(max_iter_str)

        if(b>a):
            try:
                raiz, iter, tabela, msg_erro, erro  = secante(f, a, b, toler, max_iter)
                if(erro):
                    st.error(msg_erro)
                else:
                    st.success(f'Raiz: {raiz:.6f}')
                    st.write(f'Interações: {iter}')
                    st.dataframe(tabela)

                    #Plotar um gráfico com a raiz em destaque
                    x=np.arange(a-5, b+5)
                    y = [float(f(xi)) for xi in x]

                    df = pd.DataFrame(
                        {
                            "x": x,
                            "y": y,
                        }
                    )
                    # gráfico base (linha + pontos)
                    fig = px.line(df, x="x", y="y", markers=False)

                    # adicionar o ponto interpolado, com estilo destacado
                    fig.add_scatter(
                        x=[raiz],
                        y=[0],
                        mode="markers",
                        marker=dict(size=12, color="green", symbol="circle"),
                        name="Raiz"
                    )

                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Erro: {e}")

        else:
            st.warning('Verifique o intervalo para o cálculo da raiz.')