import pickle
import numpy as np
import tensorflow as tf

'''
Utility function
Load a single pickle file containing : (Columns features, Constraints features, labels, edge indices)
'''
def load_data_from_pickle(fileName):
    fileName = fileName.numpy().decode("utf-8")
    #print(fileName+' to open')
    with open(fileName, 'rb') as f:
        loaded = pickle.load(f,encoding='latin1')        
    return loaded


'''
Utility function
Load multiple files to form one batch, and concatenate all the data
Input : File names array
Ouput : Data of the entire batch, 
        shape : tuple(cons_features , edge_indices , cols_features, labels)
'''
def load_files(files):
    #print(f"Files to load : {files}")
    if(len(files)==1):
        columns,constraints,labels,edge_indices = load_data_from_pickle(files[0])
    else:
        columns,constraints,labels,edge_indices = load_data_from_pickle(files[0])
        for i in range(1,len(files)):
            data = load_data_from_pickle(files[i])

            # Shift the edge indices !
            data[3][0,:] = data[3][0,:] + len(columns)
            data[3][1,:] = data[3][1,:] + len(constraints)
            
            # Data concatenation
            columns = np.concatenate((columns, data[0]),axis=0)
            constraints = np.concatenate((constraints, data[1]),axis=0)
            labels = np.concatenate((labels, data[2]),axis=0)
            edge_indices = np.concatenate((edge_indices, data[3]), axis=1)
            
    edge_indices[[0, 1]] = edge_indices[[1, 0]]

    # Convert numpy data types to Tensors
    labels = labels.astype(np.float32)
    cons_features = tf.convert_to_tensor(constraints, dtype= tf.float32)
    edge_indices = tf.convert_to_tensor(edge_indices,  dtype= tf.int32)
    cols_features = tf.convert_to_tensor(columns , dtype = tf.float32)
    labels = tf.convert_to_tensor( labels , dtype = tf.float32)
    

    #print('Columns features : ' + str(columns.shape))
    #print('Constraints features : '+str(constraints.shape))
    #print('Labels : ' + str(labels.shape))
    #print('Number of arcs ' + str(edge_indices.shape))
    return (cons_features , edge_indices , cols_features, labels)