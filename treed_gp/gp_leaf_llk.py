
class leaf_gp_leaf_llk:
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