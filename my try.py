import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree

from treed_gp.partitation import tree_partation
from treed_gp.gp_all_llk import leaf_gp_all_llk
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

#Create a random dataset
# rng = np.random.RandomState(1)
# X = np.sort(5 * rng.rand(24, 1), axis=0)
# y = (10*np.sin(X)*np.cos(X/2)).ravel()

# my_class = tree_partation(X,y,[[0,5]],2)
# print(my_class.X)
# print(my_class.leaf_nodes)


# my_class.return_path([0], ['no'])
# print('##################################')
# print('after using the function')
# print('leaf nodes are: ',my_class.leaf_nodes)
# print('paths are: ',my_class.record)

# leaf_nodes = my_class.leaf_nodes
# for leaf in leaf_nodes:
#     leaf_gp_temp = leaf_gp_all_llk(leaf,my_class)
#     leaf_gp_temp.train_gp()
#     lengthscale = leaf_gp_temp.lengthscale
#     boundary = leaf_gp_temp.leaf_boundary
#     print ("in leaf node {}, the lengthscale is {} and the boundary is {}".format(leaf,lengthscale,boundary))
    
    
    
######################################################################################################
# Using X,y from a GP

np.random.seed(0)
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
for leaf in leaf_nodes:
    leaf_gp_temp = leaf_gp_all_llk(leaf,my_class)
    leaf_gp_temp.train_gp()
    lengthscale = leaf_gp_temp.lengthscale
    boundary = leaf_gp_temp.leaf_boundary
    print ("in leaf node {}, the lengthscale is {} and the boundary is {}".format(leaf,lengthscale,boundary))