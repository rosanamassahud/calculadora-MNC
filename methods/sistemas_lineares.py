import numpy as np
import math

def is_symmetric(matrix):
    # Primeiro, verifique se a matriz é quadrada (condição necessária para ser simétrica)
    if matrix.shape[0] != matrix.shape[1]:
        return False
    # Verifique se a matriz original é igual à sua transposta
    # A comparação direta (matrix == matrix.T) retorna uma matriz de booleanos
    # np.array_equal verifica se todos os elementos são iguais
    return np.array_equal(matrix, matrix.T)

def residuo(A,b,x):
    r = b - np.matmul(A,x)
    return r

#função para resolução de sistema triangular inferior
#algoritmo substituições sucessivas - Livro do Frederico
def sucessiva(A, b):
    n = len(A) #ordem da matriz
    x = np.zeros(n)
    print(x)
    x[0] = b[0]/A[0][0]
    #print(x)
    for i in range(1,n):
        soma = 0
        for j in range(i):
            soma += A[i][j] * x[j]
        x[i] = (b[i]-soma)/A[i][i]
    return x

#função para resolução de sistema triangular superior
#algoritmo substituições retroativas
def retroativa(A,b):
    n = len(A)
    x = np.zeros(n)
    x[n-1] = b[n-1] / A[n-1][n-1]
    for i in range(n-2, -1,-1):
        soma = 0
        for j in range(i+1, n):
            soma += A[i][j] * x[j]
        x[i] = (b[i] - soma) / A[i][i]
    return x

def solveUpperTriangular(A,b):
    n = len(A)
    x = np.zeros(n)
    x[n-1] = b[n-1]/A[n-1][n-1]
    for k in range(n-2, -1, -1):
        soma = 0
        for j in range(k+1, n):
            soma = soma + A[k][j]*x[j]
        x[k] = (b[k]-soma)/A[k][k]
    return x

def solveLowerTriangular(A,b):
    n = len(A)
    x = np.zeros(n)

    for i in range(n):
        soma = 0
        for j in range(i):
            soma = soma + A[i][j]*x[j]
        x[i] = b[i]-soma
    return x

def LU(A,b):
    mLU = decomposicao_LU(A)
    y = solveLowerTriangular(mLU, b)
    x = solveUpperTriangular(mLU, y)
    return x

def decomposicao_LU(A):
    '''
    :param A: matriz de coeficientes
    :return: matriz decomposta
    '''
    n = len(A)
    A_copy = A.copy()
    for k in range(0, n - 1):
        for i in range(k + 1, n):
            m = float(A_copy[i][k] / A_copy[k][k]) # multiplicador
            A_copy[i][k] = m #gera um elemento da L
            for j in range(k + 1, n):
                # print(f'A[{i}][{j}]=A[{i}][{j}]-{m}*A[{k}][{j}]')
                # print(f'A[{i}][{j}]={A[i][j]}-{m}*{A[k][j]}')
                A_copy[i][j] = float(A_copy[i][j] - float(m * A_copy[k][j])) #gera elementos da U
                # print(A[i][j])
            # b_copy[i] = float(b_copy[i]) - float(m*b_copy[k])
            # print(b[i])
            # A_copy[i][k] = m
    print(A_copy)
    return A_copy

def gauss(A, b):
    '''
    :param A: matriz de coeficientes
    :param b: matriz de coeficientes independentes
    :return: solução do sistema
    '''
    #algoritmo do livro do Ruggiero
    #supor que o elemento  que está na posição akk é diferente de zero no início da etapa k
    n = len(b)
    A_copy = A.copy()
    b_copy = b.copy()
    for k in range(0,n-1):
        for i in range(k+1,n):
            m = float(A_copy[i][k]/A_copy[k][k])
            #A[i][k]=0
            for j in range(k,n):
                #print(f'A[{i}][{j}]=A[{i}][{j}]-{m}*A[{k}][{j}]')
                #print(f'A[{i}][{j}]={A[i][j]}-{m}*{A[k][j]}')
                A_copy[i][j] = float(A_copy[i][j]-float(m*A_copy[k][j]))
                #print(A[i][j])
            b_copy[i] = float(b_copy[i]) - float(m*b_copy[k])
            #print(b[i])
            A_copy[i][k] = 0
    #print(A_copy)
    #print(b_copy)
    #fase da resolução
    return retroativa(A_copy,b_copy)

def decomposicao_cholesky(A):
    '''
    :param A: matriz de coeficientes
    :return: matriz G e booleano se ocorreu Erro ou não
    '''
    n = len(A)
    G = np.zeros((n,n), dtype=float)
    for k in range(n):
        soma = 0
        for j in range(k):
            soma = soma + (G[k][j])**2
        r = A[k][k] - soma
        if(r<0):
            return [], True
        G[k][k] = r ** (1 / 2)
        for i in range(k+1, n):
            soma = 0
            for j in range(0,k):
                soma = soma + G[i][j]*G[k][j]
            G[i][k] = (A[i][k]-soma)/G[k][k]
    return G, False

def cholesky(A,b):
    '''
    :param A: matriz de coeficientes
    :param b: matriz de coeficientes independentes
    :return: vetor solução do sistema ou erro
    '''
    G, erro = decomposicao_cholesky(A)
    if(erro):
        return erro
    else:
        Gt = np.transpose(G)
        y = np.linalg.solve(G,b)
        x = np.linalg.solve(Gt,y)
        return x

def comparar(x, xk, eps):
    soma = 0
    zip_object = zip(x, xk)
    for list1_i, list2_i in zip_object:
        soma = soma + math.fabs(list1_i-list2_i)
    if (soma < eps):
        return True
    else:
        return False

def gauss_jacobi(A, b, maxiter, eps):
    n = len(b)
    sol = True
    x = b.copy()
    #calcula a solução inicial a partir de x = bi/aii
    for i in list(range(1,n+1, 1)):
        if(math.fabs(A[i-1][i-1])>0.0):
            x[i-1] = b[i-1]/A[i-1][i-1]
        else:
            sol = False
            break

    if(sol):
        print("Iteração 0")
        print("x = ", x)
        xk = x.copy()
        #maxiter = 10
        #eps = 0.01
        iter = 0

        while(iter < maxiter):
            iter = iter + 1
            for i in list(range(1,n+1,1)):
                s = 0
                for j in list(range(1,n+1,1)):
                    if((i-1) !=(j-1)):
                        s = s + A[i-1][j-1]*x[j-1]
                xk[i-1] = ((1/A[i-1][i-1])*b[i-1]-s)
            print('Iteração: ', iter)
            print('xk = ', xk)
            if (comparar(x,xk, eps)):
                x = xk.copy()
                break
            x = xk.copy()
    return x

def jacobi(A,b,maxiter, eps):
    '''
    :param A: matriz de coeficientes
    :param b: vetor de coeficientes independentes
    :param maxiter: número máximo de iterações
    :param eps: tolerância
    :return: vetor solução (x) e no. de iterações (it)
    '''
    n = len(A)
    d = np.divide(b, np.diag(A))
    x_old = d
    it = 1
    x = np.zeros(n, dtype=float)
    while(it<=maxiter):
        for i in range(n):
            s = 0
            for j in range(n):
                if(i!=j):
                    s = s + A[i][j]*x_old[j]
            s=-s+b[i]
            x[i] = s/A[i][i]
        dif = float(np.divide(np.max(np.abs(np.subtract(x,x_old))),np.max(np.abs(x))))
        if(dif<=eps):
            return x, it
        x_old = x.copy()
        print(x_old)
        it+=1
    return x, it

def seidel(A,b,maxiter, eps):
    '''
    :param A: matriz de coeficientes
    :param b: vetor de coeficientes independentes
    :param maxiter: número máximo de iterações
    :param eps: tolerância
    :return: vetor solução (x) e no. de iterações (it)
    '''
    n = len(A)
    d = np.divide(b, np.diag(A))
    x_old = d
    it = 1
    x = np.zeros(n, dtype=float)
    while(it<=maxiter):
        for i in range(n):
            s = 0
            s1=0
            for j in range(i):
                s1=s1+A[i][j]*x[j]
            s = s-s1
            s2 = 0
            for k in range(i+1,n):
                s2 = s2 + A[i][k]*x_old[k]
            s=s-s2+b[i]
            x[i] = s/A[i][i]
        dif = float(np.divide(np.max(np.abs(np.subtract(x,x_old))),np.max(np.abs(x))))
        if(dif<=eps):
            return x, it
        x_old = x.copy()
        it+=1
    return x, it

def gauss_seidel(A,b, maxiter, eps):
    n = len(b)
    sol = True
    x = b.copy()
    for i in list(range(1, n + 1, 1)):
        if (math.fabs(A[i - 1][i - 1]) > 0.0):
            x[i - 1] = b[i - 1] / A[i - 1][i - 1]
        else:
            sol = False
            break

    if (sol):
        print("Iteração 0")
        print("x = ", x)
        xk = x.copy()
        # maxiter = 10
        # eps = 0.01
        iter = 0

        while (iter < maxiter):
            iter = iter + 1
            for i in list(range(1, n + 1, 1)):
                s = 0
                for j in list(range(1, n + 1, 1)):
                    if ((i - 1) > (j - 1)):
                        s = s + A[i - 1][j - 1] * xk[j - 1]
                    elif ((i-1)<(j-1)):
                        s = s + A[i-1][j-1]*x[j-1]

                xk[i - 1] = (1 / A[i - 1][i - 1]) * (b[i - 1] - s)

            print('Iteração: ', iter)
            print('xk = ', xk)
            if (comparar(x, xk, eps)):
                x = xk.copy()
                break
            x = xk.copy()
    return x

if(__name__ == '__main__'):
    # teste para substituição sucessiva
    '''
    A = np.array([[2,0,0,0],[3,5,0,0],[1,-6,8,0],[-1,4,-3,9]])
    b = np.array([4,1,48,6])
    x = sucessiva(A, b)
    print(x)
    '''

    #teste para substituição retroativa
    '''
    A = np.array([[5,-2,6,1],[0,3,7,-4],[0,0,4,5],[0,0,0,2]])
    b = np.array([1,-2,28,8])
    x = retroativa(A, b)
    print(x)
    '''

    '''
    A = np.array([[2,-3,-1],[3,2,-5],[2,4,-1]])
    b = np.array([3,-9,-5])
    x = np.linalg.solve(A,b)
    print(x)
    '''

    #teste Gauss (fase da eliminação)
    '''
    A = np.array([[2,2,1,1],[1,-1,2,-1],[3,2,-3,-2],[4,3,2,1]], dtype=float)
    b = np.array([7,1,4,12], dtype=float)
    x= gauss(A,b)
    print('Solução: {}'.format(x))
    r = residuo(A, b, x)
    print('Residuo: {}->{}'.format(r, np.rint(r)))

    
    A = np.array([[2,3,1,5],[1,3.5,1,7.5],[1.4,2.7,5.5,12],[-2,1,3,28]], dtype=float)
    b = np.array([11,13,21.6,30], dtype=float)
    print(A)
    print(b)
    '''
    #A = np.array([[-9,5,6],[2,3,1],[-1,1,-3]], dtype=float)
    #b = np.array([11,4,-2], dtype=float)
    #print('Teste Decomposição LU')
    #x = LU(A,b)
    #print()
    #print('Solução: {}->{}'.format(x, np.rint(x)))
    #r = residuo(A,b, x)
    #print('Residuo: {}->{}'.format(r, np.rint(r)))

    #print('Gauss Seidel')
    #x = gauss_seidel(A, b, 10, 0.01)
    #print('Solução: {}->{}'.format(x, np.rint(x)))
    #r = residuo(A,b, x)
    #print('Residuo: {}->{}'.format(r, np.rint(r)))
    
    '''
    print('Decomposição Cholesky')
    #A = np.array([[4,12,-16], [12,37,-43], [-16,-43,98]], dtype=float)
    #A = np.array([[3, 2, 4], [1, 1, 2], [4, 3, -2]], dtype=float)
    #b = np.array([1,2,3], dtype=float)
    print('Exercícios de revisão. Numero 5')
    A = np.array([[20,7,9], [7,30,8], [9,8,30]], dtype=float)
    b = np.array([16,38,38], dtype=float)
    
    x = cholesky(A,b)
    if (isinstance(x, bool)):
        print('A matriz A não é definida positiva.')
    else:
        print('Solucao: ',x)
        print('Residuo: ',residuo(A,b,x))

    print('Métodos iterativos')
    #A = np.array([[5,1,1],
    #     [3,4,1],
    #     [3,3,6]])
    #b = np.array([5,6,0])
    #x = gauss_jacobi(A,b, 10, 0.05)
    #x = gauss_seidel(A,b,10,0.05)
    #x,it =jacobi(A,b,10,0.05)
    #x, it = seidel(A,b,10,0.05)
    #print('x = ', x, 'Iteracoes: ', it)
    #r = residuo(A,b,x)
    #print('residuo = ', r)

    print()
    print('Exercicios de revisão. Numero 4')
    #A = np.array([[3,1],[2,5]], dtype=float)
    #b = np.array([2,-3], dtype=float)
    A = np.array([[2,-1],[1,2]], dtype=float)
    b = np.array([1,3], dtype=float)
    x,it =jacobi(A,b,10,0.03)
    print("Resultado usando Gauss-Jacobi")
    print('x = ', x, 'Iteracoes: ', it)
    r = residuo(A,b,x)
    print('residuo = ', r)
    
    print("Resultado usando Gauss-Seidel")
    A = np.array([[10,2,6],[1,10,9],[2,-7,-10]] , dtype=float)
    b = np.array([28,7,-17], dtype=float)
    x, it = seidel(A,b,20,0.001)
    print('x = ', x, 'Iteracoes: ', it)
    r = residuo(A,b,x)
    print('residuo = ', r)
    '''
    A = np.array([[5,1,1],
         [3,4,1],
         [3,3,6]])
    b = np.array([5,6,0])
    x,it = jacobi(A,b, 10, 0.05)
    print('Resultado gauss-jacobi: ', x, 'Com ', it, 'iterações')
    residuo = residuo(A,b,x)