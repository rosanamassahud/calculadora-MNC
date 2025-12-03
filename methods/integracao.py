import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import table
'''
def f(x):
    #return (float)(1/x)
    return 2*pow(x,3)+3*pow(x,2)+6*x + 1
    #return math.sin(x)
    #return 4/(1+pow(x,2))
    #return math.log(x,math.e)
    #return 1/(x*math.log(x, math.e))
    #return math.pow(math.e, x)
'''

def pes_abs_gauss(n):
    '''
    Objetivo: calcular pesos e abscissas para
        a formula de Gauss-Legendre
    :param n: numero de pontos
    :return: pesos, abscissas e condicao de erro, sendo:
    CondErro = 0 se nao houve erro  (n>=1) e
    CondErro = 1 se n < 1
    '''
    CondErro = 0
    m = math.trunc(0.5*(n+1))
    T = np.zeros(m)
    A = np.zeros(m)
    if(n<1):
        CondErro = 1
    for i in range(m):
        z = math.cos(math.pi*((i+1)-0.25)/(n+0.5))
        while(1):
            p1 = 1
            p2 = 0
            for j in range(1,n+1):
                p3 = p2
                p2 = p1
                #polinomio de Legende no ponto z
                p1 = ((2*j-1)*z*p2-(j-1)*p3)/j
            # derivada do polinomio de Legendre no ponto z
            pp = n * (z*p1-p2)/(pow(z,2)-1)
            z1 = z
            #metodo de Newton para calcular os zeros do polinomio
            z = z1 - p1/pp
            if (abs(z-z1)< pow(10, -15)):
                break
        T[m-1-i]=z # abscissa
        A[m-1-i]=2/((1-pow(z,2))*pow(pp,2)) # peso
        # somente as raizes nao negativas sao calculadas devido a simetria
    return A, T, CondErro

def quadratura_gauss(f, a, b, n, imprime_tabela=True):
    '''
    Objetivo: Integrar uma funcao pelo metodo de Gauss-Legendre
    :param a: limite inferior
    :param b: limite superior
    :param n: numero de pontos
    :return: valor da Integral e Condicao de erro, sendo:
        CondErro = 0 se nao houve erro (n>=1) e
        CondErro = 1 se n < 1
    '''
    Integral = 0
    # calculo dos pesos e abscissas
    Avet, Tvet, CondErro = pes_abs_gauss(n)

    if(CondErro>0):
        return
    # calculo da integral

    e1 = (b-a)/2
    e2 = (a+b)/2

    if(n%2==0):
        k_ = np.arange(np.trunc(n/2),dtype=int)
        k=np.pad(k_, (int(np.trunc(n/2)), 0), 'symmetric', reflect_type='odd')
        sinal=np.ones(n)
        for s in range(int(n/2)):
            sinal[s]=-1
        #print(sinal)
    else:
        k_ = np.arange((np.trunc(n / 2) + 1), dtype=int)
        k = np.pad(k_, (int(np.trunc(n / 2)), 0), 'reflect', reflect_type='odd')
        sinal = np.sign(k)
    somas = np.zeros([n,5], dtype=float)
    for i in range(n):
        t = sinal[i] * Tvet[abs(k[i])] #np.sign(k[i]) *Tvet[abs(k[i])]
        x = e1 * t + e2
        y = f(x) # avaliar a função integrando em x
        c = Avet[abs(k[i])]
        Integral = Integral + y * c
        #print("i:{}, t:{}, x:{}, y:{}, c:{}".format(i,t,x,y,c))
        somas[i]=[i,t,x,y,c]
    #fim for

    if (imprime_tabela):
        somas_table = pd.DataFrame(somas, columns=['i', 't', 'x', 'y', 'c'])
        print(somas_table)

    Integral = e1 * Integral

    return Integral, CondErro, somas_table

def trapezio(f, a, b, n):
    x = np.linspace(a, b, n+1)
    y = [f(xi) for xi in x]
    h = (b - a) / n
    return (h/2) * (y[0] + 2*sum(y[1:-1]) + y[-1])

def trapezio_simples(x,y):
    n = len(y)
    h = x[n-1]-x[0]
    I = h/2*(y[0]+y[1])
    return I

def trapezio_repetido(y,h):
    n = len(y)
    s = 0
    for i in range(1,n-1):
        s += y[i]
    I = h/2*(y[0]+2*s+y[n-1])
    return I

def simpson_1(fx,n,h):
    '''
    :param fx: lista de valores de f(x)
    :param n: número de pontos ou o número de subintervalos
    :param h: amplitude
    :return: integral f(x) no intervalo
    '''
    if (n % 3 == 0 or n%3 == 1):
        I = 0
        I = I + fx[0]
        s = 0
        for i in range(1, int((n - 1) / 2)+1):
            s += fx[(2 * i) -1]
        s = 4 * s
        I += s
        s = 0
        for i in range(1, int((n - 1) / 2)):
            s += fx[2 * i]
        s = 2 * s
        I += s
        I += fx[n - 1]
        I = h / 3 * I
        return I
    else:
        return 'ERRO: Não foi possível realizar a aproximação com o método 1/3 de Simpson.'

def simpson_38(fx,n,h):
    '''
    :param fx: lista dos valores de f(x)
    :param n: numero de pontos
    :param h: amplitude
    :return: integral f(x) no intervalo [a,b]
    '''
    if((n-1)%3==0):
        I = 0
        I = I + fx[0]
        s = 0
        for i in range(0,int((n-1)/3)):
            #print((3*i)+1, (3*i)+2)
            s += fx[(3*i)+1] + fx[(3*i)+2]
        s = 3*s
        I += s
        s = 0
        for i in range(1,int((n-1)/3)):
            #print((3 * i))
            s += fx[3*i]
        s = 2*s
        I+=s
        I+=fx[n-1]
        I = (3*h)/8 * I
        return I
    else:
        return 'ERRO: Não foi possível realizar a aproximação com o método 3/8 de Simpson.\n' \
               '\t  O número de subintervalos deve ser múltiplo de 3.'

if (__name__ =='__main__'):
    '''
    x = [1,3]
    y = [1, 0.3333]
    integralDef = trapezio_simples(x,y)
    print('--- Trapezio simples --- ')
    print('Integral definida [1,3]: {}'.format(integralDef))
    
    x = [1,1.25,1.5,1.75,2]
    y = []
    for i in x:
        y.append(f(i))
    print(y)
    #y = [1,0.6666,0.5,0.4,0.3333]
    h = 0.25
    integralDef = trapezio_repetido(y,h)
    print('--- Trapezio com repetição --- ')
    print('Integral definida: {}'.format(integralDef))

    h = 0.16667
    x = np.arange(1,2.1,h)
    y = []
    for i in x:
        y.append(f(i))
    print(x)
    print(y)
    '''
    n=7 #NUMERO DE PONTOS
    #h=1
    y = 1
    h = 1
    integralDef = simpson_1(y,n,h)
    print('1/3 de Simpson Simples')
    print('Integral definida [1,2]: {}'.format(integralDef))
    
    '''
    x = np.arange(1,3.1,0.3333)
    y = []
    for i in x:
        y.append(f(i))
    print(x)
    print(y)
    n=7
    h=0.3333
    integralDef = simpson_1(y,n,h)
    print('1/3 de Simpson com repetição')
    print('Integral definida: {}'.format(integralDef))

    h = 0.66667
    n = 4
    x = np.arange(1,3.1,h)
    y = []
    for i in x:
        y.append(f(i))
    print(x)
    print(y)
    print('3/8 de Simpson simples')
    integralDef = simpson_38(y,n,h)
    print('Integral definida: {}'.format(integralDef))
    
    h = 0.3333
    n = 7
    x = np.arange(1,3.1,h)
    y = []
    for i in x:
        y.append(f(i))
    integralDef = simpson_38(y,n,h)
    print('3/8 de Simpson com repetição')
    print('Integral definida: {}'.format(integralDef))

    integral, erro = quadratura_gauss(1,3,2)
    print('Integral = {} Erro:{}'.format(integral, erro))
    '''
    '''
    x = np.arange(2,5.1,0.5)
    y = []
    for i in x:
        y.append(f(i))
    print(x)
    print(y)
    n=7
    h=0.5
    integralDef = simpson_1(y,n,h)
    print('1/3 de Simpson com repetição')
    print('Integral definida [2,5]: {}'.format(integralDef))
    erro_abs = abs(0.84240-integralDef)
    erro_rel = (erro_abs/0.84240)
    print("Erro absoluto = {}, Erro relativo = {}   {}%".format(erro_abs,erro_rel, erro_rel*100))

    integralDef = trapezio_repetido(y,0.5)
    print('Trapézio repetido')
    print('Integral definida [2,5]: {}'.format(integralDef))
    erro_abs = abs(0.84240-integralDef)
    erro_rel = (erro_abs/0.84240)
    print("Erro absoluto = {}, Erro relativo = {}   {}%".format(erro_abs,erro_rel, erro_rel*100))

    #  3/8 de Simpson com repetição
    h = 0.5
    n = 7
    #y = [1,0.75,0.6, 0.5, 0.4285,0.375,0.3333]
    integralDef = simpson_38(y,n,h)
    print('3/8 de Simpson com repetição')
    print('Integral definida [2,5]: {}'.format(integralDef))
    erro_abs = abs(0.84240-integralDef)
    erro_rel = (erro_abs/0.84240)
    print("Erro absoluto = {}, Erro relativo = {}   {}%".format(erro_abs,erro_rel, erro_rel*100))

    # 3/8 Simpson simples
    h = 1
    n = 4
    x = np.arange(2,5.1,1)
    y = []
    for i in x:
        y.append(f(i))
    print(x)
    print(y)
    print('3/8 de Simpson simples')
    integralDef = simpson_38(y,n,h)
    print('Integral definida [2,5]: {}'.format(integralDef))
    erro_abs = abs(0.84240-integralDef)
    erro_rel = (erro_abs/0.84240)
    print("Erro absoluto = {}, Erro relativo = {}   {}%".format(erro_abs,erro_rel, erro_rel*100))
    '''
    
    
    '''Testes para regra do trapezio simples e repetida'''
    #x = [1,3]
    #y = [1, 0.3333]
    #integralDef = trapezio_simples(x,y)
    #print('--- Trapezio simples --- ')
    #print('Integral definida [1,3]: {}'.format(integralDef))

    #y = [1, 0.6666, 0.5, 0.4, 0.3333]
    #h = 0.5
    #h = 0.2
    '''
    x = np.arange(4,5.3,0.2)
    y = []
    for i in x:
        y.append(f(i))
    print(x)
    print(y)
    
    integralDef = trapezio_repetido(y,h)
    print('--- Trapezio com repetição --- ')
    print('Integral definida [4,5.2]: {}'.format(integralDef))
    '''
    '''Testes para regra de 1/3 de Simpson'''
    #  1/3 de Simpson simples
    #h = 1
    #n = 3
    #y = [1,0.5,0.3333]
    #n = 7
    #integralDef = simpson_1(y,n,h)
    #print('1/3 de Simpson Simples')
    #print('Integral definida [4,5.2]: {}'.format(integralDef))

    #  1/3 de Simpson com repetição
    #h = 0.3333
    #n = 7
    #y = [1,0.75,0.6, 0.5, 0.4285,0.375,0.3333]
    #integralDef = simpson_1(y,n,h)
    #print('1/3 de Simpson com repetição')
    #print('Integral definida [4,5.2]: {}'.format(integralDef))
    
    '''Testes para regra de 3/8 de Simpson'''
    #  3/8 de Simpson simples
    #h = 0.66667
    #n = 4
    #y = [1,0.6,0.4285,0.3333]
    #integralDef = simpson_38(y,n,h)
    #print('3/8 de Simpson Simples')
    #print('Integral definida [1,3]: {}'.format(integralDef))

    #  3/8 de Simpson com repetição
    #h = 0.3333
    #n = 7
    #y = [1,0.75,0.6, 0.5, 0.4285,0.375,0.3333]
    #integralDef = simpson_38(y,n,h)
    #print('3/8 de Simpson com repetição')
    #print('Integral definida [1,3]: {}'.format(integralDef))

    ''' Testes com Quadratura de Gauss'''
    print("-----  Quadratura de Gauss-Legendre  -----")
    ##A, T, erro = pes_abs_gauss(4)
    ##print("A: {}".format(A))
    ##print("T:{}".format(T))
    ##print("Erro: {}".format(erro))

    ##integral, erro = quadratura_gauss(0,math.pi,6)
    ##integral, erro = quadratura_gauss(0,1,,10)
    #integral, erro = quadratura_gauss(1,5,2)
    #print('Integral = {} Erro:{}'.format(integral, erro))

    ##x = [5 * x for x in range(0, 12+1)]
    ## fx = [23,25,28,35,40,45,47,52,60,61,60,54,50]
    ##fx = [1,0.6,0.4285,0.3333]
    ##fx = [1,0.5,0.3333]
    ##print(x)
    ##print(len(fx))
    ##I = simpson_38(fx,13,1/12)
    ##I = simpson_1(fx,12,1/12)
    ##print(I)
    '''
    x=[0,0.5,1,1.5,48,48.5,49,59,69,79]
    y=[62,74,73.5,60.5,49.5,42.5,39,44.5,58,61.5]
    print(x)
    print(y)
    I1 = simpson_38(y[0:4],4,0.5)
    print("Parte 1: x[{},{}] y[{},{}] I1={}".format(x[0],x[3],y[0],y[3],I1))
    I2=trapezio_simples(x[3:5],y[3:5])
    print("Parte 2: x[{},{}] y[{},{}] I2={}".format(x[3],x[4],y[3],y[4],I2))
    I3 = simpson_1(y[4:7],3,0.5)
    print("Parte 3: x[{},{}] y[{},{}] I3={}".format(x[4],x[6],y[4],y[6],I3))
    I4 = simpson_38(y[6:10],4,10)
    print("Parte 4: x[{},{}] y[{},{}] I4={}".format(x[6],x[9],y[6],y[9],I4))
    total = I1+I2+I3+I4
    print("Total=I1+I2+I3+I4={}+{}+{}+{}={}".format(I1,I2,I3,I4,total))
    '''

    '''
    y = [0, 0.784, 0.992, 1.008, 1.216, 2]
    h = 0.4
    IntEx3 = trapezio_repetido(y,h)
    print("Resposta Ex. 3: I = {}".format(IntEx3))
    '''

    