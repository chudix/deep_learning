# Resolucion de ejercicio 2.

import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_csv('iris.csv')

def plot_scatter(dataframe):
    """ Shows a scatter matrix plot.

    It takes iris dataframe and makes a plot with x-axis as lenght and
    y-axis as width.

    Parameters
    -------
    dataframe: DataFrame
      A pandas DataFrame object from iris.csv
    """

    # plot sepal variables
    plot_1 = dataframe.plot.scatter(
        x='sepal_length',
        y='sepal_width',
        color='r',
        label='Sepal'
    )

    # plot petal variables
    plot_2 = dataframe.plot.scatter(
        x='petal_length',
        y='petal_width',
        color='b',
        label='Petal',
        ax=plot_1
    )
    # set x and y axes labels
    plot_1.set_xlabel('lenght')
    plot_1.set_ylabel('width')
    
    # show plot
    plt.show()

def plot_histogram(dataframe):
    """ Shows an histogram of iris.csv varriables

    Parameters
    -------
    dataframe: DataFrame
      A pandas DataFrame object from iris.csv
    """
    plot = dataframe.hist()
    plt.show()


plot_scatter(dataframe)
plot_histogram(dataframe)
