import numpy as np
from scipy.optimize import minimize
from treed_gp.utilities import find_path,group_data,extract_data
from treed_gp.utilities import cov_RBF,log_llk

class leaf_gp_leaf_llk:
        def __init__ (self,leaf,tree_partation):

            self.leaf = leaf
            self.tree_partation = tree_partation
            
            return None 
        
        def obtain_data(self):
            
            X = self.tree_partation.X
            y = self.tree_partation.y
            record = self.tree_partation.record
            all_boundary = self.tree_partation.boundary
            dim_record = self.tree_partation.dim_record
            split_record = self.tree_partation.split_record
            how_record = self.tree_partation.how_record

            
            path = find_path(record,self.leaf)
            X_group_path, y_group_path, self.leaf_boundary = group_data(X,y,all_boundary ,path,dim_record,split_record,how_record)
            self.X_leaf = X_group_path[-1]
            self.y_leaf = y_group_path[-1]
            
        
        def train_gp(self):
            
            self.lengthscale = optimise(self.X_leaf, self.y_leaf)
            
            
            
            

def optimise(X, y):

    opts ={'maxiter':1000,'maxfun':200,'disp': False}

    bounds=np.asarray([[0.001,10.]])

    init_two_sigma_square = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(50, 1))
    logllk_holder = [0]*init_two_sigma_square.shape[0]
    for ii,val in enumerate(init_two_sigma_square):           
        logllk_holder[ii] = log_llk(X,y,two_sigma_square=val) 
        
    x0=init_two_sigma_square[np.argmax(logllk_holder)] # we pick one best value from 50 random one as our initial value of the optimization

    # Then we minimze negative likelihood
    res = minimize(lambda x: -log_llk(X,y,two_sigma_square=x),x0,
                                bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B
    
    
    #print("estimated lengthscale",np.sqrt(res.x/2))
        
    return np.sqrt(res.x/2)  