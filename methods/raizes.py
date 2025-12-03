import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import misc

#avaliação do polinômio pelo método de horner
def horner(n, c, a):
    '''
    Objetivo: avaliar um polinomio de grau 'n' no ponto 'a'
    :param n: grau do polinomio
    :param c: lista de coeficientes
    :param a: ponto a ser avaliado
    :return: y -> ordenada (P(a))
    '''
    y = c[0]
    for i in range(1,n+1):
        y = y *a + c[i]
    return y

def limites(n,cc):
    '''
    Objetivo: achar os limites das raízes reais de uma equação polinomial
    :param n: grau do polinomio
    :param cc: lista de coeficientes, sendo P(x) = c(1)x^n+ c(2)x^(n-1)+ ... + c(n)x+c(n-1)
    :return: L (limites inferior e superior das raízes positivas e negativas, respectivamente)
    :return: msg (Mensagem)
    '''
    msg = ""
    c = np.zeros([n+1])
    for i in range(len(cc)):
        c[i] = cc[i]

    if(cc[0] == 0):
        return
    t = n
    while(True):
        #se c(n) for nulo, entao o polinomio eh deflacionado
        if(c[t]!=0):
            break
        t -= 1
        c = cc.copy()
    # calculo dos quatro limites das raizes reais
    L = np.zeros(4)
    # Verifica se todos os coeficientes são positivos
    all_coef_pos = all(c_i >= 0 for c_i in c)

    # Se forem todos positivos, troca o sinal de todos
    if all_coef_pos:
        c = [-c_i for c_i in c]
        msg = 'O polinômio não possui raízes reais positivas'
        
    for i in range(1,5):
        if(i == 2 or i == 4): # inversao da ordem dos coeficientes
            for j in range (0, int(t/2)):
                aux = c[j]
                c[j] = c[t-j]
                c[t-j] = aux
        else:
            if(i == 3): # reinversao da ordem e troca de sinais dos coeficientes
                for j in range(0, int(t/2)):
                    aux = c[j]
                    c[j] = c[t-j]
                    c[t-j] = aux
                for j in range (t-1,0,-2):
                    c[j] = -c[j]
        # se c(0) for negativo, entao eh trocado o sinal de todos os coeficientes
        if (c[0] < 0):
            for j in range(t):
                c[j] = -c[j]
        k = 1 # calculo de k, o maior indice dos coeficientes negativos
        while(1):
            if(c[k]<0 or k>t):
                break
            k+=1
        if(k==0):
            k=1
        if (k<=t):
            B = 0
            for j in range(1,t+1):
                if(c[j] < 0 and abs(c[j]) > B):
                    B = abs(c[j])
            # limite das raizes positivas de P(x) = 0 e das equacoes auxiliares
            if(((i-1)==0 or(i-1)==1) and all_coef_pos):
                L[i-1]=None
            else:    
                L[i-1] = 1 + pow((B/c[0]),(1/k))
        else:
            L[i-1] = 10**100
    L[0] = L[0]
    L[1] = 1/L[1]
    L[2] = -L[2]
    L[3] = -1/L[3]
    print(L)
    return L, msg

def ff(x):
   pass
    #fx = x**4 + 2*x**3 - 13 * x ** 2 - 14 * x + 24
    #fx = x**4 - 6 * x**3 + 42 * x + 40
    #fx = (x**3) -9*x +5
    #fx = 2 * (x ** 3) - math.cos(x + 1) - 3
    #fx = x**3 + x**2 - 10*x + 8
    #fx = 3*x**2 + 2*x - 10
    #fx = x**2 + x - 6
    # fx = x/2 - np.tan(x)
    #fx = 2* np.cos(x) -(np.e**x)/2
    #fx = x**5 - 6
    #fx = x**3 - 2*x**2 - 3*x + 10
    #fx = x**5-(10/9)*(x**3)+(5/21)*x
    
    #teste questao prova 1 2023-2
    #fx = (x**2)/2 + x*(math.log(x)-1)
    #fx = x**5-2*x**4-9*x**3+22*x**2+4*x-24
    #fx = x**3-6*x**2-x+30
    #fx = x**4-9*x**3-2*x**2+120*x-130
    #fx = x**5 -6
    #fx = x/2 - math.tan(x)
    #return fx

def troca_sinal(z, imprime_tabela=False):
    '''
    Objetivo: Achar um intervalo [a,b] onde uma função troca de sinal
    parâmetros de entrada z
        ponto a partir do qual o intervalo será gerado
    parâmetros de saída a, b, CondErro
        limite inferior e superior do intervalo e condição de erro, sendo
        CondErro = 0 se f(a)f(b) < 0 e CondErro = 1 se f(a)f(b) > 0
    '''
    if(z == 0):
        a = -0.05
        b = 0.05
    else:
        a = 0.95 * z
        b = 1.05 * z
    iter = 0
    aureo = 2/(pow(5,0.5)-1)
    Fa = ff(a)
    Fb = ff(b)
    tab = np.zeros([20,5])
    #print(iter, a, b, Fa, Fb)
    linha = [iter,a,b,Fa,Fb]
    tab[iter] = linha
    while (1==1):
        if (Fa * Fb < 0 or iter >= 20):
            break
        iter+=1
        if (abs(Fa)< abs(Fb)):
            a = a - aureo * (b-a)
            Fa = ff(a)
        else:
            b = b + aureo * (b-a)
            Fb = ff(b)
        #print(iter,a,b,Fa,Fb)
        linha = [iter,a,b,Fa,Fb]
        tab[iter] = linha
    if(Fa*Fb <=0):
        CondErro = 0
        if(imprime_tabela):
            columns = ['Iter','a','b','Fa','Fb']
            tab_table = pd.DataFrame(tab[0:iter + 1, 0:5], columns=columns)
            print(tab_table)
    else:
        CondErro = 1
    return [a,b,CondErro]

def metodo_bissecao(f, va, vb, Toler, IterMax):
    '''
    :param f: função
    :param va: valor inicial do intervalo
    :param vb: valor final do intervalo
    :param Toler: tolerância
    :param IterMax: numero maximo de iterações
    :return: raiz, iter(numero de iterações gastas), rebis_table(tabela de cálculo), msg_erro, CondErro, sendo
    CondErro = 0 se a raiz foi encontrada e CondErro = 1 em caso contrário
    '''
    msg_erro = ""
    a = va
    b = vb
    f_a = f(a)
    f_b = f(b)
    cond_erro = 0
    if (f_a * f_b > 0):
        msg_erro = 'Função não muda de sinal nos extremos do intervalor dado'
        cond_erro = 1
    
    if(not(cond_erro)):
        delta_x = abs(b - a) / 2
        iter = 1
        k = math.ceil(math.log(((b - a) / Toler), 2) - 1)
        rebis = []

        while (iter <= IterMax):
            x = (a + b) / 2
            f_x = f(x)
            rebis.append([iter,a, f_a, b, f_b, x, f_x, delta_x])
            if ((delta_x <= Toler and abs(f_x) <= Toler) or iter >= IterMax):
                break
            if ((f_a * f_x) > 0):
                a = x
                f_a = f_x
            else:
                b = x
                f_b = f_x
            delta_x = delta_x / 2
            iter += 1
        rebis_table = pd.DataFrame(rebis, columns=['iter', 'a', 'f(a)', 'b', 'f(b)', 'x', 'f(x)', 'delta_x'])

        # teste de convergência
        if (delta_x <= Toler and abs(f_x) <= Toler):
            cond_erro = 0
        else:
            cond_erro = 1
    else:
        iter = 0
        x = 0
        rebis_table = []
    return x, iter, rebis_table, msg_erro, cond_erro

def metodo_falsa_posicao(f,va, vb, Toler, IterMax):
    '''
    :param f: função
    :param va: valor inicial do intervalo
    :param vb: valor final do intervalo
    :param Toler: tolerância
    :param IterMax: numero maximo de iterações
    :return: raiz, iter(numero de iterações gastas), rebis_table(tabela de cálculo), msg_erro, CondErro, sendo
    CondErro = 0 se a raiz foi encontrada e CondErro = 1 em caso contrário
    '''
    a = va
    b = vb
    f_a = f(a)
    f_b = f(b)
    cond_erro = 0
    msg_erro = ""
    if (f_a * f_b > 0):
        msg_erro ='Função não muda de sinal nos extremos do intervalor dado'
        cond_erro = 1
    if(not(cond_erro)):
        delta_x = abs(a*f_b-b*f_a)/(f_b-f_a)
        iter = 1
        k = math.ceil(math.log(((b - a) / Toler), 2) - 1)
        rebis = []
        
        while (iter <= IterMax):
            x = (a*f_b - b*f_a) / (f_b-f_a)
            f_x = f(x)
            # print(iter, a, f_a, b, f_b, x, f_x, delta_x)
            rebis.append([iter,a, f_a, b, f_b, x, f_x, delta_x])
            if ((delta_x <= Toler and abs(f_x) <= Toler) or iter >= IterMax):
                break
            if ((f_a * f_x) > 0):
                a = x
                f_a = f_x
            else:
                b = x
                f_b = f_x
            delta_x = delta_x / (f_b-f_a)
            iter += 1
        rebis_table = pd.DataFrame(rebis, columns=['iter', 'a', 'f(a)', 'b', 'f(b)', 'x', 'f(x)', 'delta_x'])

        # teste de convergência
        if (delta_x <= Toler and abs(f_x) <= Toler):
            cond_erro = 0
        else:
            cond_erro = 1
    else:
        iter = 0
        x = 0
        rebis_table = []
    return x, iter, rebis_table, msg_erro, cond_erro

def f(x):
    return pow(x,3)-9*x+5

def g(x):
    return (x**3+5)/9

def ponto_fixo(f,g,va, vb, Toler, IterMax):
    """
    Método do ponto fixo para encontrar raízes de equações.
    :param f: função f(x)
    :param g: função g(x) tal que x = g(x)
    :param va: valor inicial
    :param vb: valor final
    :param Toler: tolerância de convergência
    :param IterMax: número máximo de iterações
    :return: raiz aproximada, número de iterações, código de erro
             (0 = convergiu, 1 = não convergiu)
    """
    x0 = (va + vb)/2 # começa pelo ponto médio do intervalo
    x = x0
    cond_erro = 0
    tab = []
    iter = 1
    msg_erro = ''
    while(iter <= IterMax):
        try:
            fx = f(x)
            x1 = g(x)
        except Exception as e:
            msg_erro = 'O método não converge com as ' \
            'informações fornecidas.{}'.format(e)
            cond_erro = 1
            break
        distancia = abs(x1 - x)
        tab.append([iter, x, fx, distancia])
        if (abs(distancia) <= Toler or abs(fx) <= Toler):
            raiz = x1
            msg_erro = ''
            cond_erro = 0
            break
        else:
            cond_erro = 1
        x = x1
        iter = iter + 1
    tabela = pd.DataFrame(tab, columns=['iter', 'x', 'f(x)', 'erro'])
    if(cond_erro):
        raiz = 0
        if(msg_erro == ''):
            msg_erro = 'Erro. O método não convergiu.'
        tabela = []
    
    return raiz, iter, tabela, msg_erro, cond_erro

def newton_raphson(f,x0, Toler, IterMax):
    '''
    :param f: função
    :param x0: valor inicial
    :param Toler: tolerância
    :param IterMax: numero maximo de iterações
    :return: raiz, numero de iterações gastas e CondErro, sendo
    CondErro = 0 se a raiz foi encontrada e CondErro = 1 em caso contrário
    '''
    Fx = f(x0) # avaliar a função em x0
    DFx = misc.derivative(f,x0,dx=1e-4) # avaliar a derivada em x0
    x = x0
    Iter = 1
    rebis = []
    rebis.append([Iter, x, DFx, Fx, 0.0])
    #print ("Iter: {}   x:{}   DFx:{}   Fx:{}".format(Iter, x, DFx, Fx, 0.0))
    while(1):
        DeltaX = -Fx/DFx
        x = x + DeltaX
        Fx = f(x) # avalia a função em x
        DFx = misc.derivative(f, x,dx=1e-4) # avalia a derivada da função em x
        Iter += 1
        rebis.append([Iter, x, DFx, Fx, DeltaX])
        #print("Iter: {}   x:{}   DFx:{}   Fx:{}   DeltaX:{}".format(Iter, x, DFx, Fx, DeltaX))
        if((abs(DeltaX) <= Toler and abs(Fx) <= Toler) or (DFx == 0) or(Iter >= IterMax)):
            break
    raiz = x
    rebis_table = pd.DataFrame(rebis, columns=["iter", "x", "f'(x)", "f(x)", "delta_x"])
    # Teste de convergencia
    if((abs(DeltaX) <= Toler) and (abs(Fx)<=Toler)):
        CondErro = 0
        msg_erro = ""
    else:
        CondErro = 1
        msg_erro = 'Erro. O método não convergiu.'
        rebis_table = []
    return raiz, Iter, rebis_table, msg_erro, CondErro

def secante(f, va,vb, Toler, IterMax):
    '''
    :param f: função
    :param va: limite inferior
    :param vb: limite superior
    :param Toler: tolerancia
    :param IterMax: numero maximo de iteracoes
    :return: raiz,
        Iter (numero de iteracoes gastas),
        CondErro (condicao de erro, sendo
            CondErro = 0 se a raiz foi encontrada
            CondErro = 1 em caso contrario
    '''
    a = va
    b = vb
    #avaliar a funcao em a e b
    Fa = f(a)
    Fb = f(b)
    if(abs(Fa) < abs(Fb)):
        t = a
        a = b
        b = t
        t = Fa
        Fa = Fb
        Fb = t
    Iter = 1
    x = b
    Fx = Fb
    rebis = []
    while(True):
        DeltaX = -Fx*((b-a)/(Fb-Fa))#-Fx/(Fb-Fa))*(b-a)
        x = x + DeltaX
        Fx = f(x) #avaliar a funcao em x
        rebis.append([Iter,a,Fa,b,Fb,x,Fx,DeltaX])

        if(((abs(DeltaX) <= Toler) and (abs(Fx)<=Toler)) or (Iter>=IterMax)):
            break
        a = b
        Fa = Fb
        b = x
        Fb = Fx
        Iter +=1

    raiz = x
    rebis_table = pd.DataFrame(rebis, columns=['Iter','a','f(a)','b','f(b)','x','f(x)','delta_x'])
    #teste de convergencia
    if((abs(DeltaX) <= Toler) and (abs(Fx) <= Toler)):
        CondErro = 0
        msg_erro = ''
    else:
        CondErro = 1
        msg_erro = 'Erro. O método não convergiu.'
        rebis_table = []
    return raiz, Iter, rebis_table, msg_erro, CondErro

#def f(x):
#    return x**3 -2*x**2+9

if __name__ == '__main__':
    #print(f(3))
    #a = 2.5
    #b = 3.0
    #Toler = 0.01
    #IterMax = 100
    #raiz, iter, tabela, msg, erro = ponto_fixo(a, b, Toler, IterMax)
    #print('Raiz: {}\nIter: {}\n{}\nMsg_erro:{}\nErro:{}'.format(raiz, iter, tabela, msg, erro))
 
    n = 4
    c = [1,2,-13,-14,24]
    L, msg = limites(n,c)
    print('--- Limites das raizes reais ---')
    print('--- Intervalo das raizes positivas: [{},{}]'.format(
        L[1],L[0]))
    print('--- Intervalo das raizes negativas: [{},{}]'.format(
        L[2], L[3]))
    print(msg)
    
    #teste questão da prova 2023-2
    #ab = troca_sinal(1)
    #print(ab)
    ##a = 0.95
    ##b = 1.64
    #a = 0.5
    #b = 1
    #Toler = 0.001
    #IterMax = 100
    #raiz, i, e = secante(2.5,3,Toler,IterMax)
    #print('Raiz: {}  Iter: {}   Erro: {}'.format(raiz,i,e))

    #testes avaliação de polinomios

    # P(x) = 3x^5 - 2x^4 + 5x^3 + 7x^2 -3x + 1
    # avaliado em x = 2
    #n = 5 # grau do polinômio
    #c = [3,-2,5,7,-3,1] # lista de coeficientes
    #a = 2 # ponto a ser avaliado
    #y = horner(n,c,a)
    #print('y = {}'.format(y))
    
    #ab = troca_sinal(-11)
    #print(ab)

    #a = ab[0] #0.5
    #a = -11
    #b = -3
    #b = ab[1] #1
    #a = 0
    #b = 1
    #Toler = 0.001
    #IterMax = 100
    #raiz, iter, tabela, msg, erro = metodo_bissecao(a, b, Toler, IterMax)
    #print('Raiz: {}\nIter: {}\n{}\nMsg_erro:{}\nErro:{}'.format(raiz, iter, tabela, msg, erro))
    #raiz = metodo_falsa_posicao(a,b,Toler,IterMax)
    #raiz, i, e = newton_raphson(a,Toler,IterMax)
    #raiz, i, e = secante(a,b,Toler,IterMax)
    #print('Raiz: {}  Iter: {}   Erro: {}'.format(raiz,i,e))

    #t = np.arange(-11, 1, 0.2)
    #yt = []
    #for i in t:
    #   yt.append(ff(i))
    #plt.plot(t, yt, 'b-')
    #plt.plot(raiz, 0, 'ro')

    #plt.grid()
    #plt.show()

    # teste derivada com funcao derivative de scipy.misc
    #print(f(2))
    #print(misc.derivative(f,2,dx=1e-4))

    #teste de Newton-Raphson

    #Toler = 0.00001
    #IterMax = 100
    #t = np.arange(-2, 6, 0.1)
    #yt = []
    #for i in t:
    #   yt.append(ff(i))
    #plt.plot(t, yt, 'b-')

    #raiz, Iter, CondErro = newton_raphson(1, Toler, IterMax)
    #raiz = metodo_bissecao(-0.75, -0.25, Toler, IterMax)
    #print('Raiz:{}   Iter:{}   CondErro:{}   '.format(raiz, Iter, CondErro))

    #raiz=metodo_bissecao(3,5,Toler, IterMax)
    #plt.plot(raiz, 0, 'ro')
    #print(raiz)
    #plt.grid()
    #plt.show()

    #print(ff(2))

    #n=3
    #c=[1,-3,-6,8]
    #n = 4
    #c = [1,2,-13,-14,24]
    #n = 6
    #c = [1, -5, 7, 19,-98,-104]
    #L = limites(n,c)
    #print(L)
    '''
    n = 5
    c = [1,-2,-9,22,4,-24]
    L = limites(n,c)
    print('--- Limites das raízes reais ---')
    print('--- Intervalo das raízes positivas: [{},{}] '.format(L[1],L[0]))
    print('--- Intervalo das raízes negativas: [{},{}] '.format(L[2], L[3]))
    t = np.arange(-5,3,0.3)
    print(t)
    y_ = []
    for i in t:
        y_.append(horner(n,c,i))
        #y_.append(ff(i))
    #print(y_)

    raiz = metodo_bissecao(2.0,2.2,0.01,100)
    print('Raiz: ',raiz)

    plt.plot(t,y_, '-b')
    plt.title('P(x)=x^4+2x^3-13x^2-14x+24')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.grid()
    plt.show()

    
    z = 5
    print('Determinação de intervalo onde ocorre troca de sinal')
    ab= troca_sinal(z,True)

    print('a =',ab[0])
    print('b =',ab[1])
    print('condErro =',ab[2])
    '''
    '''
    n = 4
    c = [1,-9,-2,120,-130]
    L = limites(n,c)
    print('--- Limites das raízes reais ---')
    print('--- Intervalo das raízes positivas: [{},{}] '.format(L[1],L[0]))
    print('--- Intervalo das raízes negativas: [{},{}] '.format(L[2], L[3]))

    a = 0.5
    b = 2
    Toler = 0.005
    IterMax = 100
    raiz = metodo_bissecao(a, b, Toler, IterMax)
    #raiz = metodo_falsa_posicao(a,b,Toler,IterMax)
    #raiz, i, e = newton_raphson(a,Toler,IterMax)
    #raiz, i, e = secante(a,b,Toler,IterMax,False)
    print('Raiz: {}'.format(raiz))
    #print('Raiz: {}  Iter: {}   Erro: {}'.format(raiz,i,e))
    '''
    '''
    t = np.arange(-1,10,0.1)
    print(t)
    y_ = []
    for i in t:
       y_.append(ff(i))

    print(y_)

    plt.plot(t,y_, '-b')
    #plt.title('f(x) = 2x^3 - cos(x+1) - 3')
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.grid()
    plt.show()
    '''
    '''
    print("Método da bissecao: ")
    raiz = metodo_bissecao(0,2,0.0001,10)
    print('Raiz = {}'.format(raiz))

    print('Método de Newton-Raphson')
    raiz, iter, erro = newton_raphson(2,0.0001, 10)
    print('Raiz: {}   Iterações: {}    Erro: {}'.format(raiz, iter, erro))
    '''
    
    #print("Método da bissecao: ")
    #raiz = metodo_bissecao(-0.4,9.6,0.0001,100)
    #print('Raiz = {}'.format(raiz))
    '''
    print('Método de Newton-Raphson')
    raiz, iter, erro = newton_raphson(3.5,0.0001, 100)
    '''