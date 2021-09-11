# Column selection for column generation
This project is the Tensorflow implementation of the machine learning part of the published paper:  

[Machine-learning–based column selection for column generation](https://pubsonline.informs.org/doi/10.1287/trsc.2021.1045). Transportation Science, 55(4):815–831,  2021.  
By Mouad Morabit, Guy Desaulniers, and  Andrea  Lodi. 

## Running the code
The project requires **tensorflow 2.0** or a more recent version, as well as Numpy and Keras.  
A sample of the data is available in the data/ folder, where each file corresponds to a bipartite graph of a column generation iteration.
You can generate your own data for the problem you want to work on and adjust the parameters in the main code!  
To launch the code, simply execute the main.py file:
```
python Main.py
```

## Citation
Please cite the paper if you use this code in your project.  
```
@article{mouad-columnselection,
author = {Morabit, Mouad and Desaulniers, Guy and Lodi, Andrea},
title = {Machine-Learning–Based Column Selection for Column Generation},
journal = {Transportation Science},
volume = {55},
number = {4},
pages = {815-831},
year = {2021},
doi = {10.1287/trsc.2021.1045}
}
```

## Questions / Bugs
Feel free to contact me if you have any questions or want to report any bugs.  
mouad.morabit@polymtl.ca
