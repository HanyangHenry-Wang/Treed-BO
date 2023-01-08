import scipy
import numpy as np
import copy
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances
from treed_gp.utilities import find_path,group_data,extract_data
from treed_gp.utilities import cov_RBF,log_llk



class leaf_gp_all_llk:

        def __init__ (self,leaf,tree_partation):

            self.leaf = leaf
            self.tree_partation = tree_partation
            
            return None
          
        
        def train_gp(self):
          
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
            
            X_extract, y_extract = extract_data(X_group_path,y_group_path)
            self.lengthscale = treegp_optimise(X_extract, y_extract)



############################## Support Functions ##########################################

def treegp_log_llk (X_extract, y_extract, two_sigma_square):
  """_summary_

  Args:
      X_extract (_type_): _description_
      y_extract (_type_): _description_
      two_sigma_square (_type_): _description_

  Returns:
      _type_: _description_
  """  
  
  depth_holder = np.array(range(len(X_extract)))
  depth_max = depth_holder[-1]

  X_extract[0].shape

  total_log_llk = 0

  for i in range(len(depth_holder)):
    weight = 2/(1+depth_max-depth_holder[i]) #since we use log likelihood, we will 'multiply' the weight instead of doing 'power'
    X_temp = X_extract[i] #if len(X_extract[i])>0 else 1
    y_temp = y_extract[i] #if len(X_extract[i])>0 else 1
    temp = weight*log_llk(X_temp, y_temp, two_sigma_square)
    total_log_llk = total_log_llk+temp

  return total_log_llk



def treegp_optimise(X_extract, y_extract):
    """_summary_

    Args:
      X_extract (_type_): _description_
      y_extract (_type_): _description_

    Returns:
      _type_: _description_
    """  

    opts ={'maxiter':1000,'maxfun':200,'disp': False}

    bounds=np.asarray([[0.001,2*10.**2]])

    init_two_sigma_square = np.random.uniform(bounds[:, 0], bounds[:, 1],size=(50, 1))
    logllk_holder = [0]*init_two_sigma_square.shape[0]
    for ii,val in enumerate(init_two_sigma_square):           
        logllk_holder[ii] = treegp_log_llk (X_extract, y_extract, val)
        
    x0=init_two_sigma_square[np.argmax(logllk_holder)] # we pick one best value from 50 random one as our initial value of the optimization

    # Then we minimze negative likelihood
    res = minimize(lambda x: -treegp_log_llk(X_extract,y_extract,two_sigma_square=x),x0,
                                bounds=bounds,method="L-BFGS-B",options=opts) #L-BFGS-B
    
    
    #print("estimated lengthscale",np.sqrt(res.x/2))
        
    return np.sqrt(res.x/2) 
