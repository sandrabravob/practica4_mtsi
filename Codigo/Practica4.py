import pandas as pd
import numpy as np
import numpy.linalg as npl


def primera_parte():
    data = pd.read_csv('data.csv', sep=',')
    file_open_column = data['Open'].tolist()

    size_no_ventana = 61

    del data

    # Creamos la matriz de r(t), r(t-1), r(t-2), ...
    matriz_dias = np.zeros((size_no_ventana, 6))

    # Creamos el vector1 como una matriz 1x1 de unos
    vector1 = np.ones((size_no_ventana, 1))

    # Calculamos r(t)
    resultados = list()
    for index in range(1, size_no_ventana):
        resultados.append((file_open_column[index] - file_open_column[index - 1]) / file_open_column[index - 1])

    # Asignamos los valores de r(t), r(t-1), r(t-2)... dejando los espacios en blancos correspondientes
    # representados como Nan

    for index_columna in range(6):
        fila = 0
        for element in resultados:
            while fila <= index_columna:
                matriz_dias[fila][index_columna] = np.NaN
                fila += 1
            try:
                matriz_dias[fila][index_columna] = element
                fila += 1
            except:
                break

    # Generamos la matriz X con sus valores
    matrizX = vector1 - matriz_dias

    # La primera columna pertenece a r(t) por lo que no es necesaria
    for index in range(6):
        matrizX = np.delete(matrizX, 0, axis=0)

    # Generamos la matriz Y como r(t)
    matrizY = matriz_dias.T[0].T

    for index in range(6):
        matrizY = np.delete(matrizY, 0, axis=0)

    for index in range(6):
        matriz_dias = np.delete(matriz_dias, 0, axis=0)

    # Aqui va el codigo que calculas las thetas, y se mete en la lista para poder acceder a ello rapido.
    thetas = matrizX.T.dot(npl.inv(matrizX.dot(matrizX.T))).dot(matrizY)
    matrizH = thetas[0] * matriz_dias.T[0] + thetas[1] * matriz_dias.T[1] + thetas[2] * matriz_dias.T[2] + thetas[3] \
              * matriz_dias.T[3] + thetas[4] * matriz_dias.T[4] + thetas[5] * matriz_dias.T[5]

    # Comparando con MatrizH
    valores1 = matrizH - matriz_dias.T[0]

    # Comparando con Matriz 0
    valores2 = 0 - matriz_dias.T[0]

    dataset = pd.DataFrame({'r(t)': matriz_dias[:, 0], 'r(t-1)': matriz_dias[:, 1], 'r(t-2)': matriz_dias[:, 2],
                            'r(t-3)': matriz_dias[:, 3], 'r(t-4)': matriz_dias[:, 4], 'r(t-5)': matriz_dias[:, 5],
                            'Prediccion': valores1, 'Paseo aleatorio': valores2}, dtype=float)
    print(dataset.to_string())

    # The next line save the result in a csv file
    # dataset.to_csv('out-result.csv', sep=',', index=False)


def segunda_parte():
    data = pd.read_csv('data.csv', sep=',')
    file_open_column = data['Open'].tolist()

    size_ventana = 81

    del data

    # Creamos la matriz de r(t), r(t-1), r(t-2), ...
    matriz_dias = np.zeros((size_ventana, 6))

    # Creamos el vector1 como una matriz 1x1 de unos. El tamaño debe ser el de la ventana
    vector1 = np.ones((30, 1))

    # Calculamos r(t)
    resultados = list()
    for index in range(1, size_ventana):
        resultados.append((file_open_column[index] - file_open_column[index - 1]) / file_open_column[index - 1])

    # Asignamos los valores de r(t), r(t-1), r(t-2)... dejando los espacios en blancos correspondientes
    # representados como Nan

    for index_columna in range(6):
        fila = 0
        for element in resultados:
            while fila <= index_columna:
                matriz_dias[fila][index_columna] = np.NaN
                fila += 1
            try:
                matriz_dias[fila][index_columna] = element
                fila += 1
            except:
                break

    for index in range(6):
        matriz_dias = np.delete(matriz_dias, 0, axis=0)

    ventana = np.zeros((30, 6))
    ventana_numero = 0
    resultados_ventana = dict()
    resultados_ventana['Pred'] = list()
    resultados_ventana['Paso'] = list()

    while ventana_numero + 30 <= size_ventana - 6:

        for index_ventana in range(30):
            ventana[index_ventana] = matriz_dias[index_ventana + ventana_numero]

        matrizX = vector1 - ventana
        matrizY = ventana.T[0].T

        thetas = matrizX.T.dot(npl.inv(matrizX.dot(matrizX.T))).dot(matrizY)
        matrizH = thetas[0] * ventana.T[0] + thetas[1] * ventana.T[1] + thetas[2] * ventana.T[2] + thetas[3] \
                  * ventana.T[3] + thetas[4] * ventana.T[4] + thetas[5] * ventana.T[5]

        # Comparando con MatrizH
        valores1 = matrizH - ventana.T[0]

        # Comparando con Matriz 0
        valores2 = 0 - ventana.T[0]

        resultados_ventana['Pred'].append(valores1[len(valores1) - 1])
        resultados_ventana['Paso'].append(valores2[len(valores2) - 1])

        ventana_numero += 1

    for index in range((size_ventana - 6) - len(resultados_ventana['Pred'])):
        matriz_dias = np.delete(matriz_dias, len(matriz_dias) - 1, axis=0)

    dataset = pd.DataFrame({'r(t)': matriz_dias[:, 0], 'r(t-1)': matriz_dias[:, 1], 'r(t-2)': matriz_dias[:, 2],
                            'r(t-3)': matriz_dias[:, 3], 'r(t-4)': matriz_dias[:, 4], 'r(t-5)': matriz_dias[:, 5],
                            'Prediccion': resultados_ventana['Pred'], 'Paseo aleatorio': resultados_ventana['Paso']},
                           dtype=float)

    print(dataset.to_string())

    # The next line save the result in a csv file
    # dataset.to_csv('out-result-ventana.csv', sep=',', index=False)


if __name__ == '__main__':
    print("Ejecutando primera parte del ejercicio")
    primera_parte()

    print('\n\n\n\n\n**********************************\n\n\n\n\n')

    print("Ejecutando segunda parte del ejercicio (Ventana Deslizante)")
    segunda_parte()
