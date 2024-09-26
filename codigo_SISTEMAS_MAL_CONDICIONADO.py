import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la matriz A y los vectores b
A = np.array([[1, 1, 1],
              [1, 1.0001, 1],
              [1, 1, 1.0001]])

b1 = np.array([3, 3.0001, 3])
# cambios en b producen grandes cambios en la solución
b2 = np.array([3, 3, 3])       

# Cálculo del determinante
det_A = np.linalg.det(A)
print("Determinante de A:", det_A)

# Cálculo de la inversa de A
inv_A = np.linalg.inv(A)
print("\nMatriz inversa de A:")
print(inv_A)

# Cálculo del número de condición
cond_A = np.linalg.cond(A)
print("\nNúmero de condición de A:", cond_A)

# Solución del sistema para ambos vectores b
x1 = np.linalg.solve(A, b1)
x2 = np.linalg.solve(A, b2)

print("\nSolución con b1:", x1)
print("Solución con b2:", x2)

# pequeños cambios en b producen grandes cambios en la solución
diff_solution = np.linalg.norm(x1 - x2)
print("\nDiferencia entre soluciones (x1 - x2):", diff_solution)

# Crear una malla para graficar los planos
x_vals = np.linspace(-1, 4, 100)
y_vals = np.linspace(-1, 4, 100)
X, Y = np.meshgrid(x_vals, y_vals)

# Definir los planos del sistema
Z1 = 3 - X - Y              
Z2 = 3 - X - 1.0001 * Y      

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar los planos
ax.plot_surface(X, Y, Z1, color='blue', alpha=0.5, rstride=100, cstride=100, label='Plano (b1)')
ax.plot_surface(X, Y, Z2, color='red', alpha=0.5, rstride=100, cstride=100, label='Plano (b2)')

# Graficar las soluciones como puntos
ax.scatter(x1[0], x1[1], x1[2], color='green', s=100, label='Solución con b1', marker='o')
ax.scatter(x2[0], x2[1], x2[2], color='purple', s=100, label='Solución con b2', marker='o')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Sistema de Ecuaciones en 3D con Soluciones')
ax.legend()
plt.show()
