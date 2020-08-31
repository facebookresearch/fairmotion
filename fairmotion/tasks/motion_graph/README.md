# Motion Graph

Motion graphs [1, 2] are used to create arbitrary motion sequences by combining fragments of sequences from a large dataset of motion clips. Our implementation loads a dataset of motion sequences, and construct a directed motion graph. In the graph, we construct an edge between two motion sequences if the tail of the first is similar to the head of the second. Motion can be generated simply by building walks on the graph.

## References
[1] Kovar, Lucas, Michael Gleicher, and Frédéric Pighin. "Motion graphs." ACM SIGGRAPH 2008

[2] Lee, Jehee, et al. "Interactive control of avatars animated with human motion data." Proceedings of the 29th annual conference on Computer graphics and interactive techniques. 2002