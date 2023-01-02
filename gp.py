import scipy
import numpy as np
import copy
from scipy.optimize import minimize
from sklearn.metrics.pairwise import euclidean_distances



class leaf_gp:
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
            
            X_extract, y_extract = extract_data(X_group_path,y_group_path)
            self.lengthscale = treegp_optimise(X_extract, y_extract)



############################## Support Functions ##########################################



def cov_RBF(x1, x2, two_sigma_square):    #    two_sigma_square = 2*\sigma^2
    """
    Firstly written by @Vu Nguyen in https://github.com/ntienvu/KnownOptimum_BO and modified by Hanyang(Henry) Wang
    
    Radial Basic function kernel (or SE kernel)
    $$
    \exp \left(-\frac{1}{2 \sigma^2}\left\|x_1-x_2\right\|^2\right)
    $$
    """

    lengthscale = two_sigma_square

    if x1.shape[1]!=x2.shape[1]:
        x1=np.reshape(x1,(-1,x2.shape[1]))

    Euc_dist=euclidean_distances(x1,x2)

    return np.exp(-np.square(Euc_dist)/lengthscale)





def log_llk(X,y,two_sigma_square):

    y_transformed =  y 

    noise_delta = 10**(-4)

    KK_x_x=cov_RBF(X,X,two_sigma_square)+np.eye(len(X))*noise_delta     
    if np.isnan(KK_x_x).any(): #NaN
        print("nan in KK_x_x !")   

    try:
        L=scipy.linalg.cholesky(KK_x_x,lower=True)
        alpha=np.linalg.solve(KK_x_x,y)

    except: # singular
        return -np.inf
    try:
        first_term=-0.5*np.dot(y_transformed.T,alpha)
        W_logdet=np.sum(np.log(np.diag(L)))
        second_term=-W_logdet

    except: # singular
        return -np.inf

    logmarginal=first_term+second_term-0.5*len(y_transformed)*np.log(2*3.1416)
    
    return logmarginal.item()




def treegp_log_llk (X_extract, y_extract, two_sigma_square):
  
  depth_holder = np.array(range(len(X_extract)))
  depth_max = depth_holder[-1]

  X_extract[0].shape

  total_log_llk = 0

  for i in range(len(depth_holder)):
    weight = 2/(1+depth_max-depth_holder[i]) #since we use log likelihood, we will 'multiply' the weight instead of doing 'power'
    #print(weight)
    X_temp = X_extract[i]
    y_temp = y_extract[i]
    temp = weight*log_llk(X_temp, y_temp, two_sigma_square)
    #print(temp)
    total_log_llk = total_log_llk+temp

  return total_log_llk



def treegp_optimise(X_extract, y_extract):

    opts ={'maxiter':1000,'maxfun':200,'disp': False}

    bounds=np.asarray([[0.001,10.]])

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




def find_path(record,node): #this function is used to find path
  for path in record:
    if node == path[-1]:
      return path
  
  
  
def group_data(X,y,boundary,path,dim_record,split_record,how_record): #this function return the X data in 'path' and the boundary in node path[-1]

    T_or_F = np.array([True]*X.shape[0])
    node_boundary = copy.deepcopy(boundary)


    X_group_path = [X]
    y_group_path = [y]

    for i in range(1,len(path)):

        node_temp = path[i]
        split = split_record[node_temp]
        dim = int(dim_record[node_temp])
        how = how_record[node_temp]

        if how == -1.0:
            T_or_F_temp = X[:,dim]<=split
            T_or_F = T_or_F*T_or_F_temp
            X_group_path.append(X[T_or_F])
            y_group_path.append(y[T_or_F])

            node_boundary[dim][1]=split

        elif how == 1.0:
            T_or_F_temp = X[:,dim]>=split
            T_or_F = T_or_F*T_or_F_temp
            X_group_path.append(X[T_or_F])
            y_group_path.append(y[T_or_F])

            node_boundary[dim][0]=split


    return X_group_path, y_group_path, node_boundary




def extract_data(X_group_path,y_group_path):
  
  X_inverse_path = X_group_path[::-1] #data size increase
  X_holder = [X_inverse_path[0]]

  y_inverse_path = y_group_path[::-1] #data size increase
  y_holder = [y_inverse_path[0]]

  for i in range(1,len(X_inverse_path)):

    X_bigger = X_inverse_path[i]
    X_smaller = X_inverse_path[i-1]
    X_diff = np.array([x for x in X_bigger if x not in X_smaller])
    X_holder.append(X_diff)
    
    y_bigger = y_inverse_path[i]
    y_smaller = y_inverse_path[i-1]
    y_diff = np.array([y for y in y_bigger if y not in y_smaller])
    y_holder.append(y_diff)

  return X_holder[::-1], y_holder[::-1]