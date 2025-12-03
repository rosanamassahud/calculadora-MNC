import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from methods.sistemas_lineares import gauss, cholesky, solveLowerTriangular, solveUpperTriangular

def mmq(x,y, imprime_tabela=False):
    '''

    :param x: lista de x
    :param y: lista de y
    :param imprime_tabela: booleano para imprimir (True) ou não(False)
                            a tabela de somatórios
    :return: valores dos coeficientes a e b
    '''
    n = len(x)
    somas = np.zeros([n,4], dtype=float)

    for i in range(0,n):
        somas[i][0]=float(x[i])
        somas[i][1]=float(y[i])
        somas[i][2]=float(x[i]*y[i])
        somas[i][3]=float(x[i]*x[i])
    s = np.einsum('ij->j', somas)
    somas = np.vstack([somas, s])
    # imprimindo a tabela de somatórios
    if (imprime_tabela):
        somas_table = pd.DataFrame(somas, columns=['x','y','xy','x^2'])
        print(somas_table)

    a = (n*somas[n][2]-somas[n][0]*somas[n][1])/(n*somas[n][3]-(somas[n][0])**2)
    b = (somas[n][0]*somas[n][2]-somas[n][1]*somas[n][3])/((somas[n][0])**2-n*somas[n][3])
    print('a={}'.format(a))
    print('b={}'.format(b))
    return [a,b]

def ajuste_linear(x,y):
    abr2 = mmq(x,y,True)
    y_ = np.sum(y)/len(y)
    sq_reg = 0
    sq_tot = 0
    for i in range(len(x)):
        sq_reg += pow(((abr2[0]*x[i]+abr2[1])-y_),2)
        sq_tot += pow((y[i]-y_),2)
    r2 = sq_reg/sq_tot
    abr2.append(r2)
    return abr2

def g_linear(ab,x):
    return ab[0]*x + ab[1]

def ajuste_exponencial(x,y):
    lny = np.log(y)
    print(lny)
    abr2 = mmq(x,lny,1)
    #abr2[1] = np.exp(abr2[1])
    y_ = np.sum(lny) / len(y)
    print('Media y = {}'.format(y_))
    sq_reg = 0
    sq_tot = 0
    for i in range(len(x)):
        sq_reg += pow(((abr2[0] * x[i] + abr2[1]) - y_), 2)
        sq_tot += pow((lny[i] - y_), 2)
    print('Soma reg: {}'.format(sq_reg))
    print('Soma tot: {}'.format(sq_tot))
    r2 = sq_reg / sq_tot
    print('R2 = {}'.format(r2))
    abr2.append(r2)
    abr2[1] = np.exp(abr2[1])
    print(abr2)
    return abr2

def g_exponencial(ab,x):
    return ab[1]*np.exp(ab[0]*x)

def ajuste_logaritmico(x,y):
    lnx = np.log(x)
    print(lnx)
    abr2 = mmq(lnx,y,1)
    y_ = np.sum(y) / len(y)
    #print('Media y = {}'.format(y_))
    sq_reg = 0
    sq_tot = 0
    for i in range(len(x)):
        sq_reg += pow(((abr2[0] * lnx[i] + abr2[1]) - y_), 2)
        sq_tot += pow((y[i] - y_), 2)
    #print('Soma reg: {}'.format(sq_reg))
    #print('Soma tot: {}'.format(sq_tot))
    r2 = sq_reg / sq_tot
    #print('R2 = {}'.format(r2))
    abr2.append(r2)

    #print(abr2)
    return abr2

def g_log(ab,x):
    return ab[0]*np.log(x)+ab[1]

def ajuste_potencia(x,y):
    lnx= np.log(x)
    lny = np.log(y)

    abr2 = mmq(lnx,lny,1)
    y_ = np.sum(lny) / len(y)
    #print('Media y = {}'.format(y_))
    sq_reg = 0
    sq_tot = 0
    for i in range(len(x)):
        sq_reg += pow(((abr2[0] * lnx[i] + abr2[1]) - y_), 2)
        sq_tot += pow((lny[i] - y_), 2)
    #print('Soma reg: {}'.format(sq_reg))
    #print('Soma tot: {}'.format(sq_tot))
    r2 = sq_reg / sq_tot
    #print('R2 = {}'.format(r2))
    abr2.append(r2)
    abr2[1] = np.exp(abr2[1])
    #print(abr2)
    return abr2

def g_potencia(ab,x):
    #print(ab)
    return ab[1]*pow(x,ab[0])

def regressao_polinomial2(x,y):
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    mx = sx/n
    my = sy/n
    sx2 = sum(x*x)
    sx3 = sum(x*x*x)
    sx4 = sum (x*x*x*x)
    sxy = sum(x*y)
    sx2y = sum(x*x*y)
    A = np.array([[n,sx,sx2],[sx,sx2,sx3],[sx2,sx3,sx4]])
    b = np.array([sy,sxy,sx2y])
    a = gauss(A,b)

def regressao_linear_EN(n,v,p,xx,yy):
    '''
    Algoritmo Frederico Campos Filho para Calcular parâmetros 
    de quadrados mínimos de modelo linear múltiplo via equações normais
    :param n: numero de pontos
    :param v: numero de variaveis
    :param p: numero de parametros
    :param x: variaveis explicativas
    :param y: variaveis respostas
    :return: b (coef. de regressao), r2 (coef. de determinacao),
    sigma2 (variancia residual) e condErro (condicao de erro)
    '''
    #print('---inicio---')
    #print(xx)
    if (v > 1 and (v + 1) != p):
        condErro = True
        return

    y = yy.copy()
    condErro = False
    # equacoes normais
    sxx = np.zeros([p, p])
    sxy = np.zeros(p)

    #regressão polinomial
    x = np.ones([n,p])
    #print(x)
    for j in range(1,p):
        for i in range(n):
            #print('--> ',xx[i][1])
            x[i][j] = pow(xx[i][1],j)
    #print(x)
    if (v == 1 and p > 2):
        for i in range(p):
            for j in range(p):
                soma = 0
                for k in range(n):
                    soma += pow(xx[k], (i + j))
                sxx[i][j] = soma
        for i in range(p):
            soma = 0
            for k in range(n):
                soma = soma + (pow(xx[k],i) * y[k])
            sxy[i] = soma
        #print(sxx)
        #print(sxy)

        b = cholesky(sxx, sxy)
        # se houve erro na resolução do sistema por cholesky
        # tenta por gauss
        if(isinstance(b, bool)):
            print('olha eu aqui')
            b = gauss(sxx,sxy)

        D = 0
        sy2 = 0
        for i in range(n):
            u = 0
            for j in range(p):
                u = u + b[j] * x[i][j]
            d = y[i] - u
            D = D + (d * d)
            sy2 = sy2 + pow(y[i], 2)
        r2 = 1 - D / (sy2 - pow(sxy[1], 2) / n)
        sigma2 = D / (n - p)
    #regressão linear múltipla
    else:
        uns = np.ones([n, 1], float)
        #print(uns)
        x = np.column_stack((uns, xx))
        #print(x)
        for i in range(p):
            for j in range(p):
                soma = 0
                for k in range(n):
                    soma = soma + x[k][i]*x[k][j]
                sxx[i][j] = soma # matriz dos coeficientes
            soma = 0
            for k in range(n):
                soma = soma + x[k][i]* y[k]
            sxy[i] = soma # vetor dos termos independentes
        #print(sxx)
        #print(sxy)

        b = cholesky(sxx, sxy)

        D = 0
        sy2 = 0
        for i in range(n):
            u = 0
            for j in range(p):
                u = u + b[j] * x[i][j]
            d = y[i] - u
            D = D + (d * d)
            sy2 = sy2 + pow(y[i], 2)
        r2 = 1 - D / (sy2 - pow(sxy[1], 2) / n)
        sigma2 = D / (n - p)

    return b, r2, sigma2, condErro

def f_reg(b, x):
    n = len(b)
    y = 0
    for i in range(n):
        if (i==0):
            y += b[i]
        else:
            y += b[i]*pow(x,i)
    return y

if (__name__=='__main__'):

    #x = [0.5,1.2,2.1,3.5,5.4]
    #y = [5.1,3.2,2.8,1.0,0.4]
    #x = [0.5,0.75,1,1.5,2.0,2.5,3.0]
    #y = [-2.8,-0.6,1,3.2,4.8,6.0,7.0]
    #x = [183,173,168,188,158,163,193,163,178]
    #y = [79,69,70,81,61,63,79,71,73]
    #plt.plot(x,y,'ro')
    #ab=ajuste_linear(x,y)
    #print('[ab] ajuste linear', ab)
    #ab = ajuste_logaritmico(x,y)
    #print('[ab] ajuste logaritmico', ab)
    #plt.show()
    '''
    gx=[]
    for i in x:
        gx.append(g_linear(ab,i))
    plt.plot(x, gx,'b--')
    plt.show()
    '''
    
    x = [1, 2, 3, 4]
    y = [3, 5, 6, 8]
    #x= [0.5, 0.75, 1, 1.5, 2.0, 2.5, 3.0]
    #y = [-2.8, -0.6, 1, 3.2, 4.8, 6.0, 7.0]
    #x = np.arange(1,9)
    #y = [0.5, 0.6, 0.9, 0.8, 1.2, 1.5, 1.7, 2.0]
    plt.plot(x, y, 'ko')
    plt.plot(x, y, 'k--')
    legenda = ['nós de interpolação', 'f(x)']
    

    print('\n\nAjuste linear')
    ab=ajuste_linear(x,y)
    t = np.arange(0, 6, 0.5)
    gx=[]
    for i in x:
        gx.append(g_linear(ab,i))
    print('R2 = {}'.format(ab[2]))
    plt.plot(x, gx, 'b-')
    legenda.append('lin(x)={}*x+{}---r2={}'.format(round(ab[0],2),round(ab[1],2), round(ab[2],2)))
    

    #print('\n\nAjuste exponencial')
    #ab = ajuste_exponencial(x, y)
    #t = np.arange(0, 9, 0.5)
    #gx = []
    #for i in x:
    #    gx.append(g_exponencial(ab, i))
    #print('R2 = {}'.format(ab[2]))
    #plt.plot(x, gx, 'm--')
    #legenda.append('exp(x)={}*e^({}*x)---r2={}'.format(round(ab[1],2),round(ab[0],2),round(ab[2],2)))

    #print('\n\nAjuste logaritmico')
    #ab = ajuste_logaritmico(x, y)
    #t = np.arange(1, 9, 0.5)
    #gx = []
    #for i in x:
    #    gx.append(g_log(ab, i))
    #print('R2 = {}'.format(ab[2]))
    #plt.plot(x, gx, 'y*-')
    #legenda.append('log(x)={}*ln(x)+{}---r2={}'.format(round(ab[0],2),round(ab[1],2),round(ab[2],2)))
    
    #print('\n\nAjuste potência')
    #abr2 = ajuste_potencia(x, y)
    ##t = np.arange(0, 9, 0.5)
    #gx = []
    #for i in x:
    #    gx.append(g_potencia(abr2, i))
    #print('R2 = {}'.format(abr2[2]))
    #plt.plot(x, gx, 'r-')
    #legenda.append('pow(x)={}*x^{}---r2={}'.format(round(abr2[1],2),round(abr2[0],2),round(abr2[2],3)))


    #plt.legend(legenda,loc="upper left")
    #plt.xlabel('x')
    #plt.ylabel('y')
    ##plt.ylim(0, 8)
    #plt.grid(True)
    #plt.show()

    #print('teste regressao linear multipla - algoritmo frederico campos')
    #print('------------------------------------------------------------')

    #n = 16
    #v = 2
    #p = 3
    #x = [[60.3, 61.1,60.2,61.2,63.2,63.6,65.0,63.8,66.0,67.9,68.2,66.5,68.7,69.6,69.3,70.6],[108.0,109.0,110.0,112.0,112.0,113.0,115.0,116.0,117.0,119.0,120.0,122.0,123.0,125.0,128.0,130.0]]
    #print(x)
    #x = np.transpose(x)
    #print(x)
    #y = [234,259,258,285,329,347,365,363,396,419,443,445,483,503,518,555]
    #print(y)
    #b, r2, sigma2, condErro = regressao_linear_EN(n,v,p,x,y)
    #print('coef. de regressao: {}\ncoef. de determinacao: {}\nvariancia residual:{}\nErro:{}'.format(b, r2, sigma2, condErro))

    #n = 11
    #v = 1
    #p = 4
    #x = [[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]]
    #x = np.transpose(x)
    ##print(x)
    #y = [0.1,0.3162,0.4472,0.5477,0.6325,0.7071,0.7746,0.8367,0.8944,0.9487,1.0]
    #b, r2, sigma2, condErro = regressao_linear_EN(n, v, p, x, y)
    #print('coef. de regressao: {}\ncoef. de determinacao: {}\nvariancia residual:{}\nErro:{}'.format(b, r2, sigma2, condErro))

    #n = 4
    #v = 1
    #p = 3
    #x = [[-1,0,1,2]]
    #x = np.transpose(x)
    #print(x)
    #y = [0,-1,0,7]
    #print(y)
    #b, r2, sigma2, condErro = regressao_linear_EN(n, v, p, x, y)

    #dados = [[1940,1950,1960,1970,1980,1991,1996,2000],
    #         [12880182,18782891,31303034,52084984,80436409,110990990,123076831,137953959],
    #         [28356133,33161506,38767423,41054053,38566297,35834485,33993332,31845211]]
    #plt.plot(dados[0],dados[1],'sg', label='urbana')
    #plt.plot(dados[0],dados[2], 'ob', label='rural')

    #plt.xlabel('ano')
    #plt.ylabel('população (milhões)')
    #plt.grid()
    #plt.show()

    #n = 8
    #v = 1
    #p = 4
    #x = [ano-1970 for ano in dados[0]]
    #y = [int(i*pow(10,-6)) for i in dados[1]]

    #x = np.transpose(x)
    #print(x)
    #print(y)
    #b, r2, sigma2, condErro = regressao_linear_EN(n, v, p, x, y)
    #print('coef. de regressao: {}\ncoef. de determinacao: {}\nvariancia residual:{}\nErro:{}'.format(b, r2, sigma2, condErro))

    #y_=[]
    #for i in x:
    #    y_.append(f_reg(b,i))
    #y_ = [i*pow(10,6) for i in y_]
    #plt.plot(dados[0],y_,'-r', label='reg poli grau 3')
    #plt.legend(loc="upper left")
    #print(dados[0])
    #print(y)
    #print(dados[1])
    #print(y_)
    #plt.show()