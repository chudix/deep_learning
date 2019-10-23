import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os

class TopologyManager(object):
    """
    """

    def __init__(self,input_shape,classes,nlayers = 3):
        self.activation_functions = ['sigmoid', 'tanh','relu']
        self._units = np.arange(1,10)
        self.topology = []
        self._layers_quantity = nlayers
        self.input_shape = input_shape
        self.classes = classes
        
    def get_random_activation(self):
        return np.random.choice(self.activation_functions)

    def get_random_units(self):
        return np.random.choice(self._units)
        
    def reset(self):
        self.__reset_topology__()
        
    def random_topology(self):
        self.__reset_topology__()
        self.topology.append(self.__input_layer__())
        for i in range(self._layers_quantity):
            self.topology.append(
                keras.layers.Dense(
                    self.get_random_units(),
                    activation=self.get_random_activation()
                    )
                )

        self.topology.append(self.__output_layer__())
        return self.topology

    def get_topology(self):
        return self.topology()

    def __repr_layer__(self,layer):
        return f'Dense({layer.units},{layer.activation.__name__})'
    
    def __reset_topology__(self):
        self.topology = []
    
    def __set_layers_quantity(self):
        n = np.random.choice(5)
        while(n == 0):
            n = np.random.choice(5)
        self._layers_quantity = n

    def __input_layer__(self):
        return keras.layers.Dense(
            self.get_random_units(),
            input_shape=self.input_shape,
            activation=self.get_random_activation()
            )

    def __output_layer__(self):
        return keras.layers.Dense(
            self.classes,
            activation='softmax'
            )
    
    
    def __repr__(self):
        layers = []
        for layer in self.topology:
            layers.append(self.__repr_layer__(layer))
        return ' - '.join(layers)
                    

class SequentialModel(object):
    '''
    Keras Sequential model.
    '''
    # Available parameters used to explore
    # differen models configurations
    lrs = np.array([0.1,0.01,0.001])
    available_epochs = np.array([50,100,200,400])
    batch_size = 32
    def __init__(self, dataset_object):
        # Dataset object
        self.dataset = dataset_object
        self.topology_manager = TopologyManager(self.__input_shape__(), self.dataset.get_n_classes())
        # after model training it holds all metrics
        self.history = None
        # None until this class setups itself
        self.model = None
        # Should be on an initialization method!
        self.__build_model__()
        self.lr = self.__get_random_lr__()
        self.epochs = self.__get_random_epochs__()


    def __build_model__(self):
        self.topology = self.topology_manager.random_topology()
        self.model = keras.Sequential(
            self.topology
        )

    def reset(self):
        '''
        should:
        - change lr
        - change model -> change topology
        - change epochs
        '''
        self.__build_model__()
        self.__reset_lr__()
        self.__reset_epochs__()
        return self

    def compile(self):
        self.model.compile(
            optimizer=keras.optimizers.SGD(self.lr),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
            )
        return self

    def fit(self):
        self.history = self.model.fit(
            self.dataset.get_x_training(),
            self.dataset.get_y_training(),
            epochs=self.epochs,
            batch_size=SequentialModel.batch_size,
            validation_data=(self.dataset.get_x_testing(), self.dataset.get_y_testing())
        )

    def run(self):
        self.compile().fit()

    def get_history(self):
        return self.history.history

    def summary(self):
        '''
        returns a dict with: lr,epocs,acc,loss,val_acc,val_loss,history
        '''
        history = self.get_history()
        return {
            'lr': self.lr,
            'epochs': self.epochs,
            'acc': history['acc'][-1],
            'loss': history['loss'][-1],
            'val_acc': history['val_acc'][-1],
            'val_loss': history['val_loss'][-1],
            'topology': self.topology_manager,
            'history': history
            }
    
    def __reset_lr__(self):
        self.lr = self.__get_random_lr__()

    def __get_random_lr__(self):
        return np.random.choice(SequentialModel.lrs)

    def __reset_epochs__(self):
        self.epochs = self.__get_random_epochs__()

    def __get_random_epochs__(self):
        return np.random.choice(SequentialModel.available_epochs)

    def __input_shape__(self):
        return (self.dataset.get_d_in(),)

    def __repr__(self):
        #change this to display all its interns as lr and epochs
        return f'Epochs:{self.epochs} - lr:{self.lr} - topology:{self.topology_manager}'
        return self.topology_manager.__repr__()


class Dataset(object):
    '''
    Data Object
    '''
    def __init__(self, dataset_name, training_ratio=0.9):
        self.dataset_name = dataset_name
        self.training_ratio = training_ratio
        self.dataset_path = os.path.abspath(os.path.join('datasets_clasificacion', self.dataset_name))
        self.data = np.loadtxt(open(self.dataset_path,"rb"),delimiter=",",skiprows=1)
        # all this variables are setted up in setup method
        # self.x_training = None
        # self.y_training = None
        # self.x_testing = None
        # self.y_testing = None
        # self.ndata = None
        # self.d_in = None
        # self.classes = None
        self.setup()

    def get_dataset(self):
        return (self.x_training, self.y_training, self.x_testing, self.y_testing)

    def get_x_training(self):
        return self.x_training

    def get_y_training(self):
        return self.y_training

    def get_x_testing(self):
        return self.x_testing

    def get_y_testing(self):
        return self.y_testing

    def get_d_in(self):
        return self.d_in

    def get_n_classes(self):
        return self.classes

    def get_n_data(self):
        return self.ndata
    
    def __split_training_testing__(self):
        '''
        Splits data into two subsets: Training and Testing
        '''
        n_training = int(self.ndata*self.training_ratio)

        self.x_training = self.x[0:n_training,:]
        self.y_training = self.y[0:n_training]
        self.x_testing = self.x[n_training:self.ndata,:]
        self.y_testing = self.y[n_training:self.ndata]
        
        return self
    
    def __dataset_dimension__(self):
        self.ndata,self.d_in=self.x.shape
        print(self.d_in)
        return self

    def __dataset_classes__(self):
        self.classes = int(self.y.max() + 1)
        return self

    def __normalize__data__(self):
        for i in range(self.d_in):
            self.x[:,i]=(self.x[:,i]-self.x[:,i].mean())/self.x[:,i].std()
        return self

    def __get_data_from_dataset__(self):
        np.random.shuffle(self.data)
        self.x,self.y = self.data[:,0:-1],self.data[:,-1]
        return self
    
    def setup(self):
        '''
        it should set up it state. ie: split dataset in testing-training,
        set n_classes, d_in... etc
        Warning: All this setup method order should not be altered
        '''
        # get data from dataset
        self.__get_data_from_dataset__()
        # set up dataset dimension
        self.__dataset_dimension__()
        # Normalize data
        self.__normalize__data__()
        # split data
        self.__split_training_testing__()
        # set dataset classes
        self.__dataset_classes__()
        return self

    def __repr__(self):
        return f'Dataset:\n\r nombre:{self.dataset_name}\nejemplos:{self.ndata}\n\r variables de entrada: {self.d_in}\n\r cantidad de clases:{self.classes}'


class ParameterExplorer(object):

    def __init__(self,dataset_name,k=2):
        self.k = k
        self.dataset = Dataset(dataset_name)
        self.models = [SequentialModel(self.dataset) for i in range(self.k)]
        #dict type with each model training summary
        self.summary = {}
#        self.plotter = HistoryPlotter(None)

    def run(self):
        '''
        it should run the k models and save its summary
        '''
        for index in range(self.k):
            # run model
            self.models[index].run()
            # save results
            self.summary[index] = self.models[index].summary()
        return self
    
    def plot_summary(self):
        ''' 
        Plot all summaries in one figure
        '''
        columns = 1
        rows = self.k

        for index in range(self.k):
            plotter = HistoryPlotter(None)
            plotter.set_history(self.summary[index]['history'])
            plt.subplot(rows,columns,index+1)
            plotter.full_plot()
            lr = self.summary[index]['lr']
            topology = self.summary[index]['topology']
            plt.title(f'lr:{lr} / topology: {topology}')
        return self
    
    def plot_summaries(self):
        '''
        Plot k summary figures
        '''
        for index in range(self.k):
            plotter = HistoryPlotter(None)
            plotter.set_history(self.summary[index]['history'])
            plotter.full_plot()
            lr = self.summary[index]['lr']
            topology = self.summary[index]['topology']
            plt.title(f'lr:{lr} / topology: {topology}')

        return self
    
class HistoryPlotter(object):

    def __init__(self, history, testing=True):
        self.history = history
        self.testing = testing
        #self.n = len(history.history['loss'])
        #self.x_axis = np.arange(self.n)

    def get_plot(self):
        return plt

    def set_history(self, history):
        self.history = history
        self.n = len(history['loss'])
        self.x_axis = np.arange(self.n)
        
        
    def __plot__(self,metric,label):
        plt.plot(self.x_axis, self.history[metric], label=label)
        plt.xlabel("Epochs")
        plt.ylabel("metric")
        plt.legend()
        plt.title("titulo")

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
        plt.figure()
        self.acc()
        self.loss()
