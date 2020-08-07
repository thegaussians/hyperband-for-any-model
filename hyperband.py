import numpy as np
import math

class hyperband:

    def __init__(self,space):
        """ The search space components, initialised by the user """

        self.search_space = space



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


    def search(self,model_definition,max_iter=81,eta=3,skip_last=1,using_loss=False):
      
        """ Assigns the number of unique brackets, number of resources and configs, implements the successive halving
        
            Arguments
              model_definition : a user defined function which contains the definition of the model
                                
              max_iter : maximum no. of iterations allowed for one hyper-parameter configuration,    dtype - int    default - 81

              eta : reduction rate of configuration in each successive halving,   dtype - int   default - 3

              skip_last : if 1 skips the last bracket and vice-versa,   dtype - int  default - 1

              using_loss : if true uses loss function of the model to evaluate in sucessive halving and if not uses val-metric of the model,   dtype - bool   default - False


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

            print('\n  Current bracket number - ', s)

            if skip_last:
                if s == 0: break

            n = int(math.ceil(int(B / max_iter / (s + 1)) * eta ** s))  # number of configurations to sample for at starting of given bracket
            r = max_iter * eta ** (-s)  # number of resources at starting for given bracket

            T = self.__sample(n)  # sampling from the search space
            metric = np.array([])

            ## this loop runs the successive halving for a given bracket
            for i in range(s + 1):

                n_i = n * eta ** (-i)  # no. of configs for given successive halving
                r_i = r * eta ** (i)  # no. of resources for given successive halving
                val_metric = np.array([model_definition(r_i,t) for t in T])  # getting the val_metric for each config
                if using_loss: T = np.array([T[i] for i in reversed(np.argsort(val_metric)[int(n / eta):])])  # implementing the successive halving
                else: T = np.array([T[i] for i in np.argsort(val_metric)[:int(n / eta)]])
                metric = np.append(metric,np.max(val_metric))

                print('\n\n \t number of reduction/successive halving done - ', i)

            best_config = np.append(best_config, T[0])  # keeping track of the best config from each bracket
            result = np.append(result,metric[-1])

        best = best_config[np.argmax(result)]
        print('\n\n the best configuration - ', best)

        return best  # return the best config from all the brackets by max val_metric

