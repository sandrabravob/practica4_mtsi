import csv, operator

# Leer el archivo 'datos.csv' con reader() y 
# mostrar todos los registros, uno a uno:

with open('ADBE.csv') as csvarchivo:
    entrada = csv.reader(csvarchivo)
    for reg in entrada:
        print(reg)  # Cada línea se muestra como una lista de campos