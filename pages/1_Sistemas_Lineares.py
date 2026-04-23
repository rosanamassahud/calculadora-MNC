import streamlit as st
import pandas as pd
import numpy as np
from methods.sistemas_lineares import gauss, LU, cholesky, residuo, jacobi, seidel, is_symmetric

st.title("Sistemas Lineares")

st.set_page_config(page_title='Sistemas Lineares', page_icon="📈", layout='centered')
metodo = st.selectbox(
    "Escolha o método para resolução de sistemas lineares",
    ("Eliminação de Gauss", "Decomposição LU", "Decomposição de Cholesky",
     "Jacobi", "Gauss-Seidel"),
)

st.write("Método escolhido:", metodo)
str_exemplo_a = "[[2,1],[5,7]]"
str_exemplo_b = "[11,13]"
if(metodo):
    if(metodo=='Eliminação de Gauss' or metodo=='Decomposição LU'):
        str_exemplo_a = "[[2,2,1,1],[1,-1,2,-1],[3,2,-3,-2],[4,3,2,1]]"
        str_exemplo_b = "[7,1,4,12]"
    elif(metodo =='Decomposição de Cholesky'):
        str_exemplo_a = '[[4,12,-16], [12,37,-43], [-16,-43,98]]'
        str_exemplo_b = '[1,2,3]'
    elif(metodo=='Jacobi' or metodo=='Gauss-Seidel'):
        str_exemplo_a = "[[5,1,1],[3,4,1],[3,3,6]]"
        str_exemplo_b = "[5,6,0]"

matriz_str = st.text_area("Digite a matriz A:", str_exemplo_a)
vetor_str = st.text_input("Digite o vetor b:", str_exemplo_b)

if(metodo=='Jacobi' or metodo=='Gauss-Seidel'):
    Toler = st.text_input("Tolerância: ", "0.05")
    Max_Iter = st.text_input("Máximo de iterações: ", "10")

bt_resolver = st.button("Resolver")

if (bt_resolver):
    resultado = []
    result_str = ""
    erro = False
    try:
        A = eval(matriz_str)
        b = eval(vetor_str)
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float)
        if(metodo == 'Eliminação de Gauss'):
            resultado = gauss(A, b)
            if(resultado is None):
                erro = True
                result_str = "Não é possível efetuar o cálculo"
                st.error('É necessário pivotação parcial para resolver esse sistema')
            else:
                result_str = "Solução: {}".format(resultado)
        elif(metodo == 'Decomposição LU'):
            resultado = LU(A,b)
            result_str = "Solução: {}".format(resultado)
        elif(metodo == 'Decomposição de Cholesky'):
            if(is_symmetric(A)):
                print("A é simétrica")
                resultado = cholesky(A,b)
                result_str = "Solução: {}".format(resultado)
            else:
                print("A não é simétrica")
                erro = True
                result_str = "Não é possível efetuar o cálculo"
                st.error('Matriz A não é simétrica')

        elif(metodo == 'Jacobi'):
            resultado, it = jacobi(A,b,int(Max_Iter),float(Toler))
            result_str = "Solução: {} com {} iterações".format(resultado, it)
        elif(metodo == 'Gauss-Seidel'):
            resultado , it = seidel(A,b,int(Max_Iter),float(Toler))
            result_str = "Solução: {} com {} iterações".format(resultado, it)
        
        if(not(erro)):
            st.success(result_str)
            vetor_residuo = residuo(A,b,resultado)
            st.warning('Resíduo: {}->{}'.format(vetor_residuo, np.rint(vetor_residuo)))
        else:
            st.error(result_str)
    except Exception as e:
        st.error(f"Erro: {e}")
