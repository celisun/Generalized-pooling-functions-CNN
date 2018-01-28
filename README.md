# Generalized Pooling Functions: mixed and gated pooling in CNN

My experiments here are based on the work done by Chen Lee, P.W.Gallagher, Z.Tu in their “Gneralizing Pooling Functions in Convolutional Neural Networks: Mixed, Gated and Tree”, 2015.

My work is to further discuss the benefits and potential of these two generalized pooling functions mixed and gated pooilng under CNN-like architectures in object recognition.

### Empirical Results

<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/table-comparison.png width="650">

It is found that the “responsive” gated pooling consistently yielded better results than “nonresponsive” mixed pooling on all datasets CIFAR10/10+/100. (one expection: the 1 per lyr mixed, which generated a surprisingly high result 64.70% outperforming both 1 per lyr gated and 1 per lyr/ch gated) This indicates that learning the mixing proportion in the way of being responsive to the features in pooling regions, although requires more additional parameters, can optimize training performances.

### Mixed / Gated Operation
<img src=https://raw.githubusercontent.com/celisun/Generalized-Pooling-Functions-Mixed-and-Gated-in-Convolutional-Neural-Networks/master/graph.png width="650">


