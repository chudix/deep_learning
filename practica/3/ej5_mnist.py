import numpy as np
from keras.datasets import mnist
from matplotlib import pyplot as plt

class Mnist:

    def __init__(self):
        # this load_data() returns a tuple in the form: ((x_train, y_train),(x_test,y_test))
        self.train_data,self.test_data = mnist.load_data()


    def __get_sample_image(self):
        n,h,w = self.train_data[0].shape
        print(f'n examples:{n}\n\r')
        #index = np.random.choice(np.arange(n))
        print(f'getting {index}th image')

    
    def view_sample_image(self):
        pass
        #plt.imshow(self.__get_sample_image())
