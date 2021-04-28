import numpy as np
import data_loader


class Model:

    def __init__(self, epochs, configurations, evaluation_method):
        
        """ write your required hyper-parameters here """

        self.learning_rate = configurations[0]
        self.epochs = epochs
        self.evaluation_method = evaluation_method
        
        

    def __model_build(self):

        """ write your model architecture here """



    def model_run(self):

        model = self.__model_build
        data = data_loader