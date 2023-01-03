import numpy as np
import copy



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