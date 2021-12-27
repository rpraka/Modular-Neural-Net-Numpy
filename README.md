# Modular Neural Network Framework with NumPy
Implementation of a modular linear neural network framework using only NumPy.

Derived from the architecture described by Rafay Khan [here.](https://medium.com/towards-artificial-intelligence/nothing-but-numpy-understanding-creating-neural-networks-with-computational-graphs-from-scratch-6299901091b0)

**Purpose**: To solidify my understanding of various ANN concepts.

**Features**: 
<ul>
    <li>Create any (N,M)-dimensional linear layers</li>
    <li>Designed to utilize vectorized operations, allowing batch and mini-batch inputs</li>
    <li>ReLU, Leaky ReLU, Tanh and Sigmoid activation layers </li>
    <li>Implemented batch/mini-batch gradient descent, momentum, rms-prop and adam optimizers</li>
    <li>Supports random-normal, Xavier and He weight intiialization strategies </li>
</ul>

**Tested on a non-linearly separable XOR binary classification problem and achieved 100% accuracy as expected.**
