import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

def plotar_grafico(df):
    # Plotar o gráfico (opcional)
    plt.plot(df['x'], df['y'], marker='o')
    plt.title('Método de Runge-Kutta de 2ª Ordem')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

# Definir a equação diferencial y' = f(x, y)
def fxy(x, y):
    return x/y - y/x

def runge_kutta_ordem_2(x0,y0,a,b,n):
    # Calcular o tamanho do passo
    h = round((b - a) / n, 4)
    
    # Vetores para armazenar os valores de x e y
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)

    # Definir os valores iniciais
    x[0] = x0
    y[0] = y0

    # Loop do Runge Kutta de 2ª ordem
    # Loop para calcular os valores de y em cada passo
    for i in range(n):
        # Calcular k1 e k2
        k1 = fxy(x[i], y[i])
        k2 = fxy(x[i] + h/2, y[i] + (h*k1)/2)
        
        # Calcular o próximo valor de y
        y[i+1] = y[i] + h * k2
        
        # Atualizar o valor de x
        x[i+1] = x[i] + h

    # Imprimir os resultados
    #for i in range(n + 1):
    #    print(f"x = {x[i]:.2f}, y = {y[i]:.4f}")

    table_result = pd.DataFrame([x,y]).pivot_table(columns=['x','y'])
    
    return h, table_result

def runge_kutta_3(x0, y0, a,b, n):
    """
    Implementa o método de Runge-Kutta de 3ª ordem para resolver uma EDO.

    Args:
        f: A função que define a EDO (f(x, y)).
        x0: Valor inicial de x.
        y0: Valor inicial de y.
        h: Tamanho do passo.
        num_iter: Número de iterações.

    Returns:
        Uma lista de tuplas (x, y) com as soluções aproximadas.
    """
    h = (b - a) / n
    #results = [[x0, y0]]
    # Vetores para armazenar os valores de x e y
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0]=x0
    y[0]=y0
    #x, y = x0, y0

    for i in range(n):
        k1 = h * fxy(x[i], y[i])
        k2 = h * fxy(x[i] + h/2, y[i] + (k1/2))
        k3 = h * fxy(x[i] + (3*h)/4, y[i] + (3*k2/4))
        
        y[i+1] = y[i] + (1/9)*(2*k1+3*k2+4*k3)
        
        x[i+1] = x[i] + h
        

    table_result = pd.DataFrame([x,y]).pivot_table(columns=['x','y'])

    return h, table_result


if(__name__=='__main__'):
    # Condições iniciais
    x0 = 1
    y0 = 2

    # Intervalo e número de passos
    a = 1  # Início do intervalo
    b = 1.2  # Fim do intervalo
    n = 2 # Número de subdivisões

    h, tabela = runge_kutta_3(x0,y0,a,b,n)
    print('h:',h)
    print(tabela)
    plotar_grafico(tabela)
