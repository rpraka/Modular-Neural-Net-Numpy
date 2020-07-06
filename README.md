# Modular Neural Network Framework with Numpy
A quick implementation of a basic modular neural network framework using only numpy.

Derived from the architechture described by Rafay Khan here:
https://medium.com/towards-artificial-intelligence/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0

**Purpose**: To solidify my understanding of various ANN concepts.

**Features**: 
<ul>
    <li>Create any (N,M)-dimensional linear layers</li>
    <li>Designed to utilize vectorized operations, allowing batch and mini-batch inputs</li>
    <li>ReLU, Leaky ReLU, Tanh and Sigmoid activation layers </li>
    <li>Implemented batch/mini-batch gradient descent, momentum, rms-prop and adam optimizers</li>
    <li>Supports random-normal, Xavier and He weight intiialization strategies </li>
</ul>

**Tested on a non-linearly separable XOR binary classification problem, with 100 percent accuracy.**
