import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import pathlib
import os
import importlib
from DataLoader import *
from BipartiteGNN import *

if __name__ == '__main__':
    # Hyperparameters
    max_epochs = 1000
    epoch_size = 16
    batch_size = 32
    test_batch_size = 32
    learning_rate = 1e-3
    early_stopping = 25
    seed = 1000
    data_path = 'dataset/'

    # Load file names in folder, and split (80%,20%) for train test respectively 
    data_files = list(pathlib.Path(data_path).glob('*.pkl'))
    data_files = [str(x) for x in data_files]
    np.random.shuffle(data_files)
    train_files = data_files[0:len(data_files)-int(len(data_files)*0.2)]
    test_files = data_files[len(data_files)-int(len(data_files)*0.2):len(data_files)]
    

    # Print hyperparameters used
    print("==== Hyperparameters ====\n")
    print(f"epoch size: {epoch_size}")
    print(f"max epochs: {max_epochs}")
    print(f"batch size: {batch_size}")
    print(f"test batch size: {test_batch_size}")
    print(f"learning rate: {learning_rate}")
    print(f"early stopping: {early_stopping}")
    print(f'Eager execution: {tf.executing_eagerly()}')
    print("\n=========================\n")
    


    print("====== Dataset Size =======\n")
    print(f"{len(train_files)} training bipartite graphs")
    print(f"{len(test_files)} test bipartite graphs")
    print("\n=========================\n")

    # Model parameters
    cons_num_features = 1
    vars_num_features = 13
    print("==== Model parameters ===\n")
    print(f"Constraints features: {cons_num_features}")
    print(f"Columns/variables features: {vars_num_features}")
    print("\n=========================\n")


    # check "load_files" functions in DataLoader.py
    # takes file name array as an input, returns data features matrices and edge indices
    FileToData_mapper = lambda x: tf.py_function(
        load_files, [x], [tf.float32, tf.int32, tf.float32, tf.float32])

    test_data = tf.data.Dataset.from_tensor_slices(test_files)
    test_data = test_data.batch(test_batch_size).map(FileToData_mapper).prefetch(1)

    # Model initialization
    model = BipartiteGNN(learning_rate = learning_rate,
        cons_num_features = cons_num_features,
        vars_num_features = vars_num_features)


    best_loss = np.inf
    rng = np.random.RandomState(seed)

    # Training loop
    for epoch in range(max_epochs + 1):
        print(f"=> Epoch {epoch}:")

        # Lazy loading of training batches
        epoch_train_files = rng.choice(train_files, epoch_size * batch_size, replace=True)
        train_data = tf.data.Dataset.from_tensor_slices(epoch_train_files)
        train_data = train_data.batch(batch_size).map(FileToData_mapper).prefetch(1)

        # Train call
        train_loss, train_accuracy, recall, tnr = model.train_or_test(train_data, train=True)
        print(f"{'Train':10} --  LOSS: {train_loss:.3f} , ACCURACY : {train_accuracy:.2f}, RECALL: {recall:.2f}, TNR: {tnr:.2f}")
        
        # Test call every n=5 epochs
        if epoch%5==0:
            test_loss, test_accuracy, recall, tnr = model.train_or_test(test_data)
            print(f"{'Test':10} --  LOSS: {test_loss:.3f} , ACCURACY : {test_accuracy:.2f}, RECALL: {recall:.2f}, TNR: {tnr:.2f}")

        # Save best model with min loss encountered so far
        if test_loss < best_loss:
            no_improv_counter = 0
            best_loss = test_loss
            model.save_state('GNN_BestWeights.pkl')
            print("=> Model saved")
        # Stop after "early_stopping" epochs without improvement
        else:
            no_improv_counter += 1
            if no_improv_counter % early_stopping == 0:
                print(f"No improvement after {no_improv_counter} consecutive epochs.")
                print("/!\ Early stopping")
                break

    # Load best model found and print test metrics
    model.restore_state('GNN_BestWeights.pkl')
    test_loss, test_accuracy, recall, tnr = model.train_or_test(test_data)
    print(f"Saved model -- LOSS: {test_loss:.3f}, ACCURACY : {test_accuracy:.2f}, RECALL: {recall:.2f}, TNR: {tnr:.2f}")
