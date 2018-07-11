# Generalized Pooling Functions: mixed and gated pooling in CNN

Experiments are implementing and are based on Chen Lee, P.W.Gallagher, Z.Tu, Gneralizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated and Tree, 2015. This work was used to satisfy the coursework final project for the research course Machine Learning Practial in University of Edinburgh

Experiment results again show the advantages obtained from these innovate pooling functions (mixed and gated) in CNN-like architectures in object recognition. Scripts show part of my implementation

### Empirical Results

<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/table-comparison.png width="650">

It is found that the “responsive” gated pooling consistently yielded better results than “nonresponsive” mixed pooling on all datasets CIFAR10/10+/100. (one expection: the 1 per lyr mixed, which generated a surprisingly high result 64.70% outperforming both 1 per lyr gated and 1 per lyr/ch gated) This indicates that learning the mixing proportion in the way of being responsive to the features in pooling regions, although requires more additional parameters, can optimize training performances.

### Mixed / Gated Operation
<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/graph.png width="650">


