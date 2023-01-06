import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

from treed_gp.partitation import tree_partation
from treed_gp.gp_all_llk import leaf_gp_all_llk
from treed_gp.gp_leaf_llk import leaf_gp_leaf_llk
import GPy

# 2 dimensional example
# rng = np.random.RandomState(1)
# X = np.array([[0.1,1.2],[1.1,0.9],[1.3,1.9],[0.5,2.3],[1.8,0.7],[2.5,1.1],[2.7,2.7],[1.9,2.4]])  #the boundary is [[0,3],[0,3]]
# y = np.array([1, 1.2, 2.1, 2.2, 3.3, 3.4, 4.3,4.5]).ravel()

# my_class = tree_partation(X,y,[[0,3],[0,3]],2)
# print(my_class.X)
# print(my_class.leaf_nodes)


# my_class.return_path([0], ['no'])
# print('##################################')
# print('after using the function')
# print('leaf nodes are: ',my_class.leaf_nodes)
# print('paths are: ',my_class.record)

# leaf_nodes = my_class.leaf_nodes
# for leaf in leaf_nodes:
#     leaf_gp_temp = leaf_gp(leaf,my_class)
#     leaf_gp_temp.train_gp()
#     lengthscale = leaf_gp_temp.lengthscale
#     boundary = leaf_gp_temp.leaf_boundary
#     print ("in leaf node {}, the lengthscale is {} and the boundary is {}".format(leaf,lengthscale,boundary))
    


#########################################################################################################
# 1 dimensional case

# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(24, 1), axis=0)
y = (10*np.sin(X)*np.cos(X/2)).ravel()

my_class = tree_partation(X,y,[[0,5]],2)
print(my_class.X)
print(my_class.leaf_nodes)


my_class.return_path([0], ['no'])
print('##################################')
print('after using the function')
print('leaf nodes are: ',my_class.leaf_nodes)
print('paths are: ',my_class.record)

leaf_nodes = my_class.leaf_nodes
for leaf in leaf_nodes:
    leaf_gp_temp = leaf_gp_all_llk(leaf,my_class)
    leaf_gp_temp.train_gp()
    lengthscale = leaf_gp_temp.lengthscale
    boundary = leaf_gp_temp.leaf_boundary
    print ("in leaf node {}, the lengthscale is {} and the boundary is {}".format(leaf,lengthscale,boundary))
    
    
    
#####################################################################################################
#Using X,y from a GP

np.random.seed(50)
rng = np.random.RandomState(1)
X_sample1=np.array([[0]])
Y_sample1=np.array([[0]])
X = np.sort(5 * rng.rand(24, 1), axis=0)
kernel1 = GPy.kern.RBF(input_dim=1,variance=1,lengthscale=1.5)
m1 = GPy.models.GPRegression(X_sample1,Y_sample1,kernel1)
m1.Gaussian_noise.variance.fix(0.0)
y = m1.posterior_samples_f(X.reshape(-1,1),size=1)
y = y.reshape(-1,) 

my_class = tree_partation(X,y,[[0,5]],2)
print(my_class.X)
print(my_class.leaf_nodes)


my_class.return_path([0], ['no'])
print('##################################')
print('after using the function')
print('leaf nodes are: ',my_class.leaf_nodes)
print('paths are: ',my_class.record)

leaf_nodes = my_class.leaf_nodes
print('Here is the result of all likelihood!')
for leaf in leaf_nodes:
    leaf_gp_temp = leaf_gp_all_llk(leaf,my_class)
    leaf_gp_temp.train_gp()
    lengthscale = leaf_gp_temp.lengthscale
    boundary = leaf_gp_temp.leaf_boundary
    print ("in leaf node {}, the lengthscale is {} and the boundary is {}".format(leaf,lengthscale,boundary))

print('Here is the result of leaf likelihood!')
for leaf in leaf_nodes:
    leaf_gp_temp = leaf_gp_leaf_llk(leaf,my_class)
    leaf_gp_temp.obtain_data()
    leaf_gp_temp.train_gp()
    lengthscale = leaf_gp_temp.lengthscale
    boundary = leaf_gp_temp.leaf_boundary
    print ("in leaf node {}, the lengthscale is {} and the boundary is {}".format(leaf,lengthscale,boundary))
    # print(leaf_gp_temp.X_leaf)
    # print(leaf_gp_temp.y_leaf)





# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.tree import plot_tree
# from treed_gp.partitation import tree_partation
# from treed_gp.gp_all_llk import leaf_gp_all_llk
# from treed_gp.gp_leaf_llk import leaf_gp_leaf_llk
# import GPy

# import torch
# #from botorch.fit import fit_gpytorch_mll
# from botorch.models import SingleTaskGP
# from botorch.test_functions import Hartmann
# from gpytorch.mlls import ExactMarginalLogLikelihood
# from botorch.fit import fit_gpytorch_mll 
# from botorch.acquisition import ExpectedImprovement
# from botorch.optim import optimize_acqf
# import matplotlib.pyplot as plt
# from botorch.models import FixedNoiseGP, ModelListGP

# from botorch.test_functions import Ackley
# from botorch.test_functions import Beale
# from botorch.test_functions import Branin
# from botorch.test_functions import EggHolder


# from botorch.test_functions import synthetic
# #from __future__ import annotations
# import math
# from typing import List, Optional, Tuple
# import torch
# from botorch.test_functions.base import BaseTestProblem
# from torch import Tensor

# class Paper_function(synthetic.SyntheticTestFunction):

#     dim = 2
#     _bounds = [(-2., 6.), (-2., 6.)]
#     # _optimal_value = -1.0
#     # _optimizers = [(0.0, 0.0)]
#     _check_grad_at_opt = False

#     def evaluate_true(self, X: Tensor) -> Tensor:
#         x1, x2 = X[..., 0], X[..., 1] 
#         part1 = torch.exp(-x1**2-x2**2)
#         return x1*part1
    
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dtype = torch.double

# from gpytorch.kernels import MaternKernel, RBFKernel, IndexKernel
# from gpytorch.means import ZeroMean


# total_record_BO_zeromean_all_llk = []

# for exp in range(1):

#   #print(exp)
#   torch.manual_seed(exp)

#   ackley = Branin(negate=True) # I name my function ackley because the first function I tried is ackley
#   a = 15*(torch.rand(20, 1,device=device, dtype=dtype))-5. #change boundary here !!!
#   print('a is ',a)
#   b = 15*(torch.rand(20, 1,device=device, dtype=dtype))  ##change boundary here !!!
#   train_x = torch.column_stack((a, b))
#   train_obj = ackley(train_x).unsqueeze(-1) 

#   best_value = train_obj.max()   
#   best_value_holder = [best_value] 


#   for j in range (20):

#     print('steps: ',j)

#     best_value = best_value_holder[-1]

#     choice = [] 
#     acq_val = [] 

#     train_x = train_x.numpy()
#     train_obj = train_obj.numpy()

#     my_tree = tree_partation(train_x,train_obj,[[-5.,10.],[0.,15.]],2)  #change boundary here !!!
#     my_tree.return_path([0], ['no'])

#     leaf_nodes = my_tree.leaf_nodes

#     for leaf in leaf_nodes:
#       leaf_gp_temp = leaf_gp_all_llk(leaf,my_tree)
#       leaf_gp_temp.train_gp()

#       train_X_temp = torch.tensor(leaf_gp_temp.X_leaf)
#       train_obj_temp = torch.tensor(leaf_gp_temp.y_leaf)
#       boundary_temp = leaf_gp_temp.leaf_boundary
#       train_yvar = torch.tensor(10**(-4), device=device, dtype=dtype)


#       lengthscale = leaf_gp_temp.lengthscale
#       boundary = leaf_gp_temp.leaf_boundary
#       print ("in leaf node {}, the lengthscale is {} and the boundary is {}".format(leaf,lengthscale,boundary))
      
#       covar_module =  RBFKernel()  #define my own kernel here
#       model = FixedNoiseGP(train_X_temp, train_obj_temp, train_yvar.expand_as(train_obj_temp),mean_module = ZeroMean(),covar_module=covar_module).to(device)
#       model.covar_module.lengthscale = torch.tensor([lengthscale]).reshape(1,1)
#       mll = ExactMarginalLogLikelihood(model.likelihood, model) .to(device)

#   #     try:
#   #       fit_gpytorch_mll(mll) 
#   #     except:
#   #       pass

#       #find the next evaluation
#       EI = ExpectedImprovement(model=model, best_f=best_value) .to(device)  

#       new_point_analytic, val = optimize_acqf(
#           acq_function=EI,
#           bounds=torch.tensor(boundary_temp,device=device, dtype=dtype).T,  
#           q=1,
#           num_restarts=20,
#           raw_samples=100,
#           options={},
#       ) 


#       choice.append(new_point_analytic)
#       acq_val.append(val)

#       print('the lengthscale of model is: ',model.covar_module.lengthscale )
#       print('*********************')


#     idx = np.argmax(np.array(acq_val))
#     new_point_analytic = choice[idx]

#     new_obj = ackley(new_point_analytic).unsqueeze(-1) .to(device)

#     train_x = torch.tensor(train_x)
#     train_obj = torch.tensor(train_obj)
    
#     train_x = torch.cat((train_x, new_point_analytic))
#     train_obj = torch.cat((train_obj, new_obj))
#     best_value = train_obj.max()
#     best_value_holder.append(best_value)


#   best_value_holder = np.array(best_value_holder)
#   total_record_BO_zeromean_all_llk.append(best_value_holder)