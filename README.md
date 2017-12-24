# Generalized Pooling Functions: Mixed&Gated Pooling in Convolutional Neural Networks

### Empirical Results

<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/table-comparison.png width="650">

we see that the “responsive” gated pooling consistently yielded better results than “nonresponsive” mixed pooling on all datasets CIFAR10/10+/100. (one expection: the 1 per lyr mixed, which generated a surprisingly high result 64.70% outperforming both 1 per lyr gated and 1 per lyr/ch gated) This indicates that learning the mixing proportion in the way of being responsive to the features in pooling regions, although requires more additional parameters, can optimize training performances.

### Related Work
<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/graph.png width="650">

The "mixed" pooling operation can be expressed as the following:

<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/formula-mixed.png width="650">

The "gated" pooling operation can be expressed as the following:

<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/formula-gated.png width="600">
