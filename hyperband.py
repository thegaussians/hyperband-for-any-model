import numpy as np
import math,model_definition,json,os


class hyperband:

    def __init__(self, train_path, val_path, checkpoint_path, **kwargs):
        """ The search space components, initialised by the user """

        self.train_path = train_path
        self.validation_path = val_path
        self.checkpoint_path = checkpoint_path
        if kwargs: self.search_space = kwargs
        else: self.search_space = self.__build_search_space()   


    def __build_search_space(self):
   
        space_params = json.load(open('hyper-parameters.json'))
        space = {}
        for key,value in space_params.items():
            if value['scale'] == 'log_scale': space[key] = np.logspace(value['min'],value['max'],(abs(value['max']-value['min'])/value['step']))
            else: space[key] = np.arange(value['min'],value['max'],value['step'])
        return space
        


    def __sample(self,n_samples):
        """ It draws a sample 'n_samples' times from the search space uniformly

            Arguments 
              n_samples : no. of points to sample from the search space
                        dtype - int

            Returns
              config : sampled hyperparameter configurations
                        dtype - numpyarray        
        """

        config = np.array([np.random.choice(self.search_space[val],n_samples,replace=True) for val in self.search_space.keys()])
        return np.transpose(config)



    def __checkpoint_update(self,sample_results,experiments):
        """ removing the model checkpoints of eliminated configs after each succ. halving """
        
        for key,value in experiments.items():
            if key not in sample_results: 
                os.remove(self.checkpoint_path+str(value)+'.h5')
                experiments.pop(value)
        return experiments



    def search(self, model_definition, max_iter = 81, eta = 3, skip_last = False, evaluation_method = 1):
      
        """ Assigns the number of unique brackets, number of resources and configs, implements the successive halving
        
            Arguments
              model_definition : a user defined function which contains the definition of the model
                                
              max_iter : maximum no. of iterations allowed for one hyper-parameter configuration,    dtype - int    default - 81

              eta : reduction rate of configuration in each successive halving,   dtype - int   default - 3

              skip_last : if True skips the last bracket and vice-versa,   dtype - bool  default - False

              evaluation_method : if 0 uses validation loss of the model to evaluate in sucessive halving and if not uses validation metric of the model,   dtype - bool   default - 1


            Returns
              best : the best hyper-parameter configuration 
        """

        logeta = lambda x: math.log(x) / math.log(eta)
        s_max = int(logeta(max_iter))
        B = (s_max + 1) * max_iter

        
        result = np.array([])
        best_config = np.array([])

        ## this loop denotes the no. of unique run of successive halving
        for s in reversed(range(s_max + 1)):

            print('\n\n    Current bracket number - ', s)

            if skip_last:
                if s == 0: break

            n = int(math.ceil(int(B / max_iter / (s + 1)) * eta ** s))  # number of configurations to sample for at starting of given bracket
            r = max_iter * eta ** (-s)  # number of resources at starting for given bracket

            T = self.__sample(n)  # sampling from the search space
            experiments =  dict(map(lambda x,y: (y,x), np.arange(T.shape[0]), T))  # creating experiment ids for each samples at each new brackets

            ## this loop runs the successive halving for a given bracket
            for i in range(s + 1):

                n_i = n * eta ** (-i)  # no. of configs for given successive halving
                r_i = r * eta ** (i)  # no. of resources for given successive halving

                val_metric = np.array([model_definition(r_i,t,evaluation_method,experiments[t]) for t in T])  # getting the val_metric for each config
                if using_loss: T = np.array([T[i] for i in reversed(np.argsort(val_metric)[int(n / eta):])])  # implementing the successive halving
                else: T = np.array([T[i] for i in np.argsort(val_metric)[:int(n / eta)]])
                #metric = np.append(metric,np.max(val_metric))
                experiments = self.__checkpoint_update(T,experiments)
                print('\n \t number of reduction/successive halving done - ', i)

            best_config = np.append(best_config, T)  # keeping track of the best config from each bracket
            result = np.append(result,val_metric)

        best = best_config[np.argmax(result)]
        print('\n\n the best configuration - ', best)

        return best  # return the best config from all the brackets by max val_metric


