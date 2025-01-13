import numpy as np

#Matrice initiale 

matrice = np.array([[12, 45, 78,89], [56,78, 98, 123], [34, 56, 78,89],[45,57,89,100] , [23,45,67,78],[78,98,123,145],[56,78,98,123],[34,56,78,89]])   



x_min = np.min(matrice)
x_max = np.max(matrice)
normalized_matrice =  (matrice - x_min ) / (x_max - x_min)

print(normalized_matrice)
