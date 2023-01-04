import numpy as np
import copy
from sklearn.tree import DecisionTreeRegressor

class tree_partation:
    
    def __init__ (self,X,y,boundary,max_depth):
        self.X = X
        self.y = y
        self.boundary = boundary
        self.max_depth = max_depth
        self.tree = regression_tree (X,y,max_depth)
         
        self.n_nodes = self.tree.tree_.node_count
        
        self.record = []
        self.leaf_nodes = []

        self.dim_record = np.zeros(self.n_nodes)-10
        self.split_record = np.zeros(self.n_nodes)
        self.how_record = np.zeros(self.n_nodes)
         
        return None 
     
        
        
    def return_path(self, sofar_path, sofar_split):
        
        tree_ = self.tree.tree_
        node = sofar_path[-1]
        print(sofar_path)

        self.record.append(sofar_path)

        #what is the splitting point?
        left_node = tree_.children_left[node]
        if left_node>0:
            dim = tree_.feature[node]
            split = tree_.threshold[node]
            new_split = new_split_point(self.X, dim, split)


        #left_node = tree_.children_left[node]
        if left_node==-1:  #check whether the node is a leaf node
            print ("node {} is leaf-node".format(node))
            print()
            self.leaf_nodes.append(node)

        
        if left_node>0:
            print ("in node {}: dimension {} is smaller than or equal to {}".format(tree_.children_left[node],dim,new_split))
            print ("in node {}: dimension {} is larger than or equal to {}".format(tree_.children_right[node],dim,new_split))
            self.split_record[tree_.children_left[node]] = new_split
            self.split_record[tree_.children_right[node]] = new_split
            self.how_record[tree_.children_left[node]] = -1
            self.how_record[tree_.children_right[node]] = 1
            self.dim_record[tree_.children_left[node]] = dim
            self.dim_record[tree_.children_right[node]] = dim

        #left one
        left_node = tree_.children_left[node]
        if left_node>0:
            left_path = copy.deepcopy(sofar_path)
            left_split = copy.deepcopy(sofar_split)

            left_path.append(left_node)
            left_split.append(split)
            self.return_path(left_path, left_split)


        #right one
        right_node = tree_.children_right[node]
        if right_node>0:
            right_path = copy.deepcopy(sofar_path)
            right_split = copy.deepcopy(sofar_split)

            right_path.append(right_node)
            right_split.append(split)

            self.return_path(right_path, right_split)
            
             




############################## Support Functions ##########################################

def regression_tree (X,y,max_depth):
    
    regr_1 = DecisionTreeRegressor(max_depth=max_depth,min_samples_leaf=3)
    regr_1.fit(X, y)
    
    return regr_1



def new_split_point(X, dimension, split):

    data_dim = X[:,dimension]
    difference_array = np.absolute(data_dim-split)
    index = np.argmin(difference_array)
      
    new_split = data_dim[index]
     
    return new_split

