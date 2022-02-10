# Motion Graph

Motion graphs [1, 2] are used to create arbitrary motion sequences by combining fragments of sequences from a large dataset of motion clips. Our implementation loads a dataset of motion sequences, and construct a directed motion graph. In the graph, we construct an edge between two motion sequences if the tail of the first is similar to the head of the second. Motion can be generated simply by building walks on the graph.

## References
[1] Jehee Lee, Jinxiang Chai, Paul S. A. Reitsma, Jessica K. Hodgins, and Nancy S. Pollard. 2002. Interactive control of avatars animated with human motion data. ACM Trans. Graph. 21, 3 (July 2002), 491–500. DOI:https://doi.org/10.1145/566654.566607

[2] Lucas Kovar, Michael Gleicher, and Frédéric Pighin. 2002. Motion graphs. ACM Trans. Graph. 21, 3 (July 2002), 473–482. DOI:https://doi.org/10.1145/566654.566605
