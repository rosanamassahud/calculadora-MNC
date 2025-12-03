import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import factorial

def p2(x):
    return 1.0-(7/3)*x+(2/3)*x*x

def lagrange(x,y,xx):
    '''
    :param x: lista de xi
    :param y: lista de yi
    :param xx: valor a ser interpolado
    :return: valor interpolado e booleano se houve erro
    '''
    n = len(x)  # número de amostras
    if len(y) != n:
        print("x e y devem ter o mesmo tamanho")
        return 0, True
    soma = 0  # Inicializa o somatório com 0
    for i in range(n):  # Calcula o somatório (range gera uma sequência de números)
        produto = 1
        d = 1
        for j in range(n):  # Calcula o produtório
            if (i != j):  # Se i diferente de j
                produto = produto * (xx - x[j])
                d = d * (x[i] - x[j])
        soma = soma +y[i] *(produto/d)
    return soma, False

def metodo_newton(x, y, xx, imprime_tabela=True):
    '''
    :param x: lista de xi
    :param y: lista de yi
    :param xx: valor a ser interpolado
    :param imprime_tabela: booleano para indicar se imprime ou nao a tabela de diferenças divididas
    :return: valor interpolado
    '''
    n = len(x)
    difd = np.zeros([n, n])
    # difd[:,0]=y[:]

    for i in range(n):
        difd[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            difd[i, j] = (difd[i + 1][j - 1] - difd[i][j - 1]) / (x[i + j] - x[i])

    # imprimindo a tabela de diferenças divididas
    if (imprime_tabela):
        difd_table = pd.DataFrame(difd)
        print(difd_table)

    # cálculo do termo interpolado (f(z))
    xterm = 1
    yint = difd[0][0]
    for order in range(1, n):
        xterm = xterm * (xx - x[order - 1])
        yint = yint + difd[0][order] * xterm
    if(imprime_tabela==False):
        difd_table = []
    return yint,difd_table

def metodo_gregory_newton(x,y,xx, imprime_tabela=True):
    '''
    :param x: abscissas
    :param y: ordenadas
    :param xx: valor a interpolar
    :param imprime_tabela: imprimir (ou não) a tabela de interpolação
    :return: valor interpolado
    '''
    n = len(x)
    diff = np.zeros([n, n])
    for i in range(n):
        diff[i][0] = y[i]
    #construcao para diferencas finitas
    for k in range(n):
        for i in range(n-1,k,-1):
            diff[i][k+1]=diff[i][k]-diff[i-1][k]
    # imprimindo a tabela de diferenças finitas
    if (imprime_tabela):
        difd_table = pd.DataFrame(diff)
        print(difd_table)

    #avaliacao do polinomio pelo processo de Horner
    u = (xx - x[0])/(x[1]-x[0]) #(x-x0)/h
    r = diff[0,0]
    for i in range(1,n):
        sum = diff[i][i]/factorial(i)
        for j in range(0, i):
            sum = sum * (u-j)
        r = r + sum
    
    if(imprime_tabela==False):
        difd_table = []

    return r, difd_table

if (__name__ == '__main__'):
    x = [1.0,1.3,1.7,2.0]
    y = [0.8415,1.2526,1.6858,1.8186]
    xi = 1.5
    yi, erro = lagrange(x,y,xi)
    if(erro):
        print('ERRO!')
    else:
        print('Valor interpolado: ', yi)

    '''
    x = [-1.0,0.0,2.0]
    fx = [4.0,1.0,-1.0]
    xi = 1
    
    #x = [-50, -5, 5, 75]
    #fx = [-300, -50, 180, 350]
    #xi = 0
    yi, erro = lagrange(x,fx,xi)
    if(erro):
        print('ERRO!')
    else:
        print('Valor interpolado: ', yi)
    gx = []
    plt.plot(x,fx,'b-')
    t = np.arange(-5,5,0.5)
    #print(t)
    for i in t:
        gx.append(p2(i))

    plt.plot(t, gx,'r--')
    plt.plot(xi,p2(xi), 'gs')
    plt.title('Demonstração de Interpolação polinomial c/ resolução de sistema linear')
    #plt.ylim(-2,10)
    plt.xlabel('x')
    plt.legend(['f(x)', 'p2(x)', 'valor interpolado'])
    plt.show()
    '''
    '''
    x = [-1.0, 0.0, 2.0]
    y = [4.0, 1.0, -1.0]
    z = 1 # valor a ser interpolado
    valor, erro = lagrange(x,y,z)
    print("Valor interpolado é: {}. Erro: {}".format(valor,erro))
    #valor = metodo_newton(x, y, z, imprime_tabela=True)
    #print("Valor interpolado é: ", valor)
    #valor = metodo_gregory_newton(x,y,z,imprime_tabela=True)
    #print("Valor interpolado é: ", valor)

    #t = np.arange(x[0]-25, x[len(x)-1]+25, 5)
    #yt = []
    #for i in t:
    #    yt.append(metodo_newton(x, y, i, False))
    #plt.plot(t, yt, 'b-')
    #plt.plot(x, y, 'ro')
    #plt.plot(z, valor, 'g*')
    #plt.grid()
    #plt.show()
    '''