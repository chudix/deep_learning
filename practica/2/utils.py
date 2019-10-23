import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def equals_vector(x, y):
    return np.all(x==y)

def verificar_igualdad(x,y):
    iguales=equals_vector(x, y)
    if iguales:
        print("Los vectores x e y son iguales:")
    else:
        print("Los vectores x e y son distintos:")

    print("x: ", x)
    print("y: ", y)


def plot_regresion_lineal_univariada(w,b,x,y,title=""):
    # genero una ventana de dibujo con una sola zona de dibujo (1,1)
    f,ax_data=plt.subplots(1,1)
    # dibujo el conjunto de datos como pares x,y y color azul
    ax_data.scatter(x, y, color="blue")
    # establezco el titulo principal
    f.suptitle(f"{title}")
    # Dibujo la recta dada por los parámetros del modelo (w,b)
    x_pad = 10
    min_x, max_x = x.min() - x_pad, x.max() + x_pad
    ax_data.plot([min_x, max_x], [min_x * w + b, max_x * w + b], color="red",label=f"w={w:.5f}, b={b:.5f}")
    # agrego una leyenda con la etiqueta del parametro `label`
    ax_data.legend()
    # Establezco las etiquetas de los ejes x e y
    ax_data.set_xlabel("x (Horas estudiadas)")
    ax_data.set_ylabel("y (Nota)")



def plot_regresion_lineal(w,b,x,y,title=""):
    # genero una ventana de dibujo con una sola zona de dibujo (1,1)
    # que permita graficos en 3D
    figure = plt.figure(figsize=(10, 10), dpi=100)
    ax_data = figure.add_subplot(1, 1, 1, projection='3d')

    #dibujo el dataset en 3D (x1,x2,y)
    x1=x[:,0]
    x2=x[:,1]
    ax_data.scatter(x1,x2, y, color="blue")
    figure.suptitle(title)

    # Dibujo el plano dado por los parametros del modelo (w,b)
    # Este codigo probablemente no sea facil de entender
    # si no tenes experiencia con calculos en 3D
    detail = 0.05
    # genero coordenadas x,y de a pares, las llamo xx e yy
    xr = np.arange(x.min(), x.max(), detail)
    yr = np.arange(y.min(), 10, detail)
    xx, yy = np.meshgrid(xr, yr)
    # calculo las coordenadas z en base a xx, yy, y el modelo (w,b)
    zz = xx * w[0] + yy * w[1] + b
    # dibujo la superficie dada por los puntos (xx,yy,zz)
    surf = ax_data.plot_surface(xx, yy, zz, cmap='Reds', alpha=0.5, linewidth=0, antialiased=True)

    # Establezco las etiquetas de los ejes
    ax_data.set_xlabel("x1 (Horas estudiadas)")
    ax_data.set_ylabel("x2 (Promedio)")
    ax_data.set_zlabel("y (Nota)")
    # Establezco el titulo del grafico
    ax_data.set_title("(Horas estudiadas x Promedio) vs Nota")





# imprime los puntos para un dataset bidimensional junto con la frontera de decisión del modelo
def plot_regresion_logistica2D(modelo, x, y,title="",detail=0.1,subplot=211):

    assert x.shape[1]==2,f"x debe tener solo dos variables de entrada (tiene {x.shape[1]})"
    # nueva figura
    plt.figure()
    # gráfico con la predicción aprendida
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, detail),
                         np.arange(y_min, y_max, detail))

    Z = np.c_[xx.ravel(), yy.ravel()]

    Z = modelo.predict(Z)
    Z = Z.argmax(axis=1)  # para Keras
    titulo = f"{title}: regiones de cada clase"
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3)  # ,  cmap='RdBu')
    plt.colorbar()
    plt.title(titulo)


    # puntos con las clases
    plt.scatter(x[:, 0], x[:, 1], c=y)

def dividir_dataset_training_testing(dataset, training_ratio=0.8):
    '''
    Toma como parametro un dataset y devuelve los 
    correspondientes conjuntos de datos de entrenamiento
    y testing.
    
    :param dataset:dataset del modelo
    :type dataset: np.array like
    :param training_ratio: proporcion del training respecto al dataset entero
    :type training_ratio: Float

    :return: tupla de tipo (x_training, y_training, x_test, y_test)
    '''
    n,d_in=dataset.shape

    n_training = int(n*training_ratio)
    n_test = n - n_training

    np.random.shuffle(dataset)
    x,y = dataset[:,0:-1],dataset[:,-1]

    return (x[0:n_training,:],y[0:n_training],x[n_training:n,:],y[n_training:n])

def plot_loss_history(history, testing=True):
    '''
    Toma un history como parametro y grafica el loss del modelo

    :param history: historial del entrenamiento del modelo
    :type history: history

    '''
    plt.figure()

    n = len(history.history['loss'])
    x_axis = np.arange(n)
    
    plt.plot(x_axis,history.history['loss'], label="training")
    if testing:
        plt.plot(x_axis,history.history['val_loss'], label="testing")

    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("loss")

class MetricsPlotter(object):

    def __init__(self, history, testing=True):
        self.history = history
        self.testing = testing
        self.n = len(history.history['loss'])
        self.x_axis = np.arange(self.n)

    def __plot__(self,metric,label):
        plt.plot(self.x_axis, self.history.history[metric], label=label)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend()

    def loss(self,standalone=False):
        self.__plot__("loss","loss_training")
        if self.testing and not standalone:
            self.__plot__("val_loss","loss_testing")

    def acc(self, standalone=False):
        self.__plot__("acc","acc_training")
        if self.testing and not standalone:
            self.__plot__("val_acc","acc_testing")

    def acc_vs_loss(self):
        self.acc(standalone=True)
        self.loss(standalone=True)

    def full_plot(self):
        plt.figure(figsize=(10,10))
        self.acc()
        self.loss()

