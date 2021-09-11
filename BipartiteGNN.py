import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from sklearn.metrics import confusion_matrix
from DataLoader import *
import pickle
import pathlib
import os
import importlib
import sys


'''
    The main class representing a bipartite graph with two types of nodes : constraints and variables/columns nodes
    The class implements the GNN operations described in the paper.
'''
class BipartiteGNN(K.Model):

    '''
    Initialization of the different modules and attributes
    Attributes : 
    - embedding_size : Embedding size for the intermediate layers of the neural networks
    - cons_num_features : Number of constraint features, the constraints data matrix expected has the shape (None,cons_num_features)
    - vars_num_features : Number of variable features, the variables data matrix expected has the shape (None,vars_num_features)
    - learning_rate : Optimizer learning rate
    - activation : Activation function used in the neurons
    - initializer : Weights initializer
    '''
    def __init__(self, embedding_size = 32, cons_num_features = 1, 
        vars_num_features = 13, learning_rate = 1e-3, 
        activation = K.activations.relu, initializer = K.initializers.Orthogonal):
        super(BipartiteGNN, self).__init__()


        self.embedding_size = embedding_size
        self.cons_num_features = cons_num_features
        self.vars_num_features = vars_num_features
        self.learning_rate = learning_rate
        self.activation = activation
        self.initializer = initializer()
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate) 

        # constraints embedding layer
        self.cons_embedding = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # variables/columns embedding layer
        self.var_embedding = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
        ])

        # NN responsible for the intermediate updates
        self.join_features_NN = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer)
        ])

        # Representations updater for the constraints, called after the agregation
        self.cons_representation_NN = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),  
        ])
        # Representations updater for the variables/columns, called after the agregation
        self.vars_representation_NN = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),  
        ])

        # NN for final output, i.e., one unit logit output
        self.output_module = K.Sequential([
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.initializer),
            K.layers.Dense(units=1, activation=None, kernel_initializer=self.initializer)
        ])

        # Build of the input shapes of all the NNs
        self.build()

        # Order set for loading/saving the model
        self.variables_topological_order = [v.name for v in self.variables]


    '''
    Build function, sets the input shapes. Called during initialization
    '''
    def build(self):
        self.cons_embedding.build([None, self.cons_num_features])
        self.var_embedding.build([None, self.vars_num_features])
        self.join_features_NN.build([None, self.embedding_size*2])
        self.cons_representation_NN.build([None, self.embedding_size*2])
        self.vars_representation_NN.build([None, self.embedding_size*2])
        self.output_module.build([None, self.embedding_size])
        self.built = True

    '''
    Main function taking as an input a tuple containing the three matrices :
    - cons_features : Matrix of constraints features, shape : (None, cons_num_features)
    - edge_indices : Edge indices linking constraints<->variables, shape : (2, None)
    - vars_features : Matrix of variables features, shape : (None, vars_num_features)

    Output : logit vector for the variables nodes, shape (None,1)
    '''
    def call(self, inputs):
        cons_features, edge_indices, vars_features = inputs

        # Nodes embedding, constraints and variables
        cons_features = self.cons_embedding(cons_features)
        vars_features = self.var_embedding(vars_features)

        # ==== First Pass : Variables -> Constraints ====

        # compute joint representations
        joint_features = self.join_features_NN(
                tf.concat([
                    tf.gather(
                        cons_features,
                        axis=0,
                        indices=edge_indices[0])
                    ,
                    tf.gather(
                        vars_features,
                        axis=0,
                        indices=edge_indices[1])
                ],1)
        )

        # Aggregation step
        output_cons = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[0], axis=1),
            shape=[cons_features.shape[0], self.embedding_size]
        )
        # Constraints representations update
        output_cons = self.cons_representation_NN(tf.concat([output_cons,cons_features],1))



        # ==== Second Pass : Constraints -> Variables ====

        # compute joint representations
        joint_features = self.join_features_NN(
                tf.concat([
                    tf.gather(
                        output_cons,
                        axis=0,
                        indices=edge_indices[0])
                    ,
                    tf.gather(
                        vars_features,
                        axis=0,
                        indices=edge_indices[1])
                ],1)
        )

        # Aggregation step
        output_vars = tf.scatter_nd(
            updates=joint_features,
            indices=tf.expand_dims(edge_indices[1], axis=1),
            shape=[vars_features.shape[0], self.embedding_size]
        )
        # Variables representations update
        output_vars = self.vars_representation_NN(tf.concat([output_vars,vars_features],1))

        # ==== Final output from the variables representations (constraint nodes are ignored)
        output = self.output_module(output_vars)

        return output


    '''
    Save model and current weights to a given path
    '''
    def save_state(self, path):
        with open(path, 'wb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                pickle.dump(v.numpy(), f)

    '''
    Load an existing model from a given path
    '''
    def restore_state(self, path):
        with open(path, 'rb') as f:
            for v_name in self.variables_topological_order:
                v = [v for v in self.variables if v.name == v_name][0]
                v.assign(pickle.load(f))


    '''
    Training/Test function
    Input: 
    - data : a batch of data, type : tf.data.Dataset
    - train: boolean, True if function called for training (i.e., compute gradients and update weights),
                False if called for test
    Output:
    tuple(Loss, Accuracy, Recall, TNR) : Metrics
    '''
    def train_or_test(self, data, train=False):
        mean_loss = 0
        mean_accuracy = 0
        confusion_mat = np.zeros((2,2))
        batches_counter = 0
        for batch in data:
            cons_features, edge_indices, vars_features, labels = batch
            input_tuple = (cons_features, edge_indices, vars_features)
      
            # When called train=True, compute gradient and update weights
            if train:
                with tf.GradientTape() as tape:
                    # Get logits from the bipartite GNN model
                    logits = self(input_tuple)
                    # Compute loss, class weight of 10 to deal with the data imbalance
                    loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels,logits=logits,pos_weight=10)

                # Compute gradient and update weights
                grads = tape.gradient(target=loss, sources=self.variables)
                self.optimizer.apply_gradients(zip(grads, self.variables))
            # If no optimizer instance set, no training is performed, give outputs and metrics only
            else:
                logits = self(input_tuple)
                loss = tf.nn.weighted_cross_entropy_with_logits(labels=labels,logits=logits,pos_weight=10)

            # Sigmoid, 0 to 1 output
            prediction = tf.round(tf.nn.sigmoid(logits))
            correct_pred = tf.equal(prediction, labels)
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

            loss = tf.reduce_mean(loss)

            # Batch loss, accuracy, confusion matrix
            mean_loss += loss
            batches_counter += 1 
            confusion_mat += confusion_matrix(labels, prediction)

        # Batch average loss
        mean_loss /= batches_counter
        
        # Compute Recall, TNR and Balanced accuracy from the confusion matrix
        tnr = confusion_mat[0][0] / (confusion_mat[0][0]+confusion_mat[0][1])
        recall = confusion_mat[1][1] / (confusion_mat[1][0]+confusion_mat[1][1])
        balanced_acc = (tnr+recall)/2

        return (mean_loss, balanced_acc, recall, tnr)