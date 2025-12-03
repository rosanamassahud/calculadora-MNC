import numpy as np
import pandas as pd



#def fxy(x,y):
    #pass
    #return x - 2 * y + 1
    #return (x**2-y**2)/(x*y)

def euler(fxy, a,b,m,x0, y0):
    '''
    :param fxy: funcao em (x,y)->y'
    :param a: limite inferior
    :param b: limite superior
    :param m: número de subintervalos
    :param y0: valor inicial
    :return: h e tabela solução do PVI
    '''
    h = (b-a)/m
    x = x0
    y = y0
    VetX = []
    VetY = []
    #avaliar f(x,y) em x = x0 e y = y0
    Fxy = fxy(x,y) 
    VetX.append(x)
    VetY.append(y)
    tab = np.zeros([m+1,4], dtype=float)
    tab[0] = [0, x, y, Fxy]
    for i in range(1,m+1):
        x = a + i * h
        y = y + h * Fxy
        #avaliar f(x,y) em x = xi e y = yi
        Fxy = fxy(x,y)
        tab[i] =[i,x,y,Fxy]
        VetX.append(x)
        VetY.append(y)
    table = pd.DataFrame(tab, columns=['i', 'x', 'y', 'Fxy'])
    #print(table)
    return h, table

def runge_kutta_2(fxy, x0, y0, a, b, n):
    # Calcular o tamanho do passo
    h = (b - a) / n
    
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

def runge_kutta_3(fxy, x0, y0, a, b, n):
    """
    Implementa o método de Runge-Kutta de 3ª ordem para resolver uma EDO.

    Args:
        fxy: A função que define a EDO (f(x, y)).
        x0: Valor inicial de x.
        y0: Valor inicial de y.
        a: Valor inicial do intervalo
        b: Valor final do intervalo
        n: Número de subdivisões.

    Returns:
        Tamanho do passo (h) e a tabela de resultados (x, y) com as soluções aproximadas.
    """
    h = (b - a) / n
    #results = [[x0, y0]]
    # Vetores para armazenar os valores de x e y
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0]=x0
    y[0]=y0

    for i in range(n):
        k1 = h * fxy(x[i], y[i])
        k2 = h * fxy(x[i] + h/2, y[i] + (k1/2))
        k3 = h * fxy(x[i] + (3*h)/4, y[i] + (3*k2/4))
        
        y[i+1] = y[i] + (1/9)*(2*k1+3*k2+4*k3)
        
        x[i+1] = x[i] + h
        

    table_result = pd.DataFrame([x,y]).pivot_table(columns=['x','y'])

    return h, table_result

if __name__ == '__main__':
    y, x, tabela = euler(1,2,10, 2)
    print('y: ', y)
    print('x: ', x)
    print(tabela)