# Resolucion de ejercicio 2.

import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('iris.csv')

# Plot scatter matrix
plot1 = pd.plotting.scatter_matrix(dataframe)
# Variables histogram are in scatter matrix. If we want to plot them
# separately:
#plot2 = dataframe.hist()

# Show plotting
plt.show()
