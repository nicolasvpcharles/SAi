# SeAi
A Simple Artificial Inteligence module writen in c++ that has all the basics to have a small neuronal network on your project

<br>(see some <a href="https://github.com/nicolasvpcharles/footballpredictor">project examples</a>) <br>

---

The network contains:
- An input layer.
- One hidden layer with a configurable number of neurons.
- One output neuron.

The hidden layer uses the `tanh` activation function. The output layer uses the `sigmoid` activation function, producing a prediction between 0 and 1.

Training uses backpropagation and gradient descent.

## Requirements

- A C++ compiler that supports C++11 or later.
- The C++ Standard Library.

No external libraries are required.

## Installation

Copy `neuralNetwork.h` into your project directory and include it in your source file:

```cpp
#include "neuralNetwork.h"
```

## Creating a Neural Network

Create an instance of the `neuralNetwork` class using three main parameters and the input data:

```cpp
#include "neuralNetwork.h"

int main()
{
    vector<vector<double>> inputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    vector<double> answers = {
        0, 1, 1, 0
    };

    neuralNetwork network(inputs, answers, 8, 2);

    return 0;
}
```

The constructor takes four arguments:

| Parameter | Description |
|---|---|
| `nInputlayer` | Training input data. Each inner vector represents one sample. |
| `nAwnsers` | Expected output for each input sample. |
| `numberNeurons` | Number of neurons in the hidden layer. |
| `nNeuronDepth` | Stored as a configuration value; it does not currently control the network architecture. |

Each input sample must have the same number of features, and the number of expected answers must match the number of input samples.

## Training the Network

Use the `train()` method to train the network:

```cpp
network.train(10000, 0.01);
```

Parameters:

- `eproach`: Number of training epochs. Defaults to `1000`.
- `learningRate`: Controls the size of weight updates. Defaults to `0.01`.

More epochs can improve learning, but excessive training may lead to overfitting. The appropriate settings depend on the dataset.

Training initializes the weights and biases randomly. Calling `train()` again starts a new training session rather than continuing from the existing parameters.

## Making Predictions

After training, use `predict()` to obtain a prediction:

```cpp
double result = network.predict({1, 0});

cout << result << endl;
```

The result is a number between 0 and 1.

For binary classification, use `round()` to convert the prediction into a Boolean value:

```cpp
bool classification = network.round(result);

cout << classification << endl;
```

The classification rule is:

- `true` if the prediction is greater than or equal to `0.5`.
- `false` if the prediction is below `0.5`.

The meaning of `true` and `false` depends on how the training labels are defined.

## Saving the Network

Use `save()` to save the network's parameters:

```cpp
network.save("model");
```

The target directory must already exist and be writable.

The current implementation saves two files:

- `weight.txt`: Weights connecting the input layer to the hidden layer.
- `bias.txt`: Biases of the hidden layer.

**Important:** The current `save()` method does not save `outputWeight` or `outputBias`. Therefore, these files alone are not sufficient to restore a fully trained network.

## Loading Parameters

Use the following methods to load the saved parameters:

```cpp
network.loadWeight("model");
network.loadBias("model");
```

These methods load the hidden-layer weights and biases from the specified directory.

**Important:** The output-layer parameters are not loaded. To make predictions using a previously trained model, you must also preserve and restore `outputWeight` and `outputBias`. The current module does not implement this yet.

## Available Methods

| Method | Description |
|---|---|
| `train(eproach, learningRate)` | Initializes and trains the network. |
| `predict(input)` | Returns a prediction between 0 and 1. |
| `round(prediction)` | Converts a prediction into a Boolean classification. |
| `save(filePath)` | Saves hidden-layer weights and biases. |
| `loadWeight(filePath)` | Loads hidden-layer weights. |
| `loadBias(filePath)` | Loads hidden-layer biases. |
| `randomWeight()` | Generates a random weight between -1 and 1. |
| `tanhActivation(x)` | Applies the hyperbolic tangent activation function. |
| `sigmoid(x)` | Applies the sigmoid activation function. |
| `loss(prediction, answer)` | Calculates the squared error for one prediction. |
| `gradientDescent(weight, learningRate, gradient)` | Performs one gradient descent weight update. |
| `multiplyMatrices(A, B)` | Multiplies two compatible matrices. |
| `tanhDerivative(x)` | Calculates the derivative of tanh, assuming `x` is an activated value. |
| `sigmoidDerivative(x)` | Calculates the derivative of sigmoid, assuming `x` is an activated value. |
| `ReLU(x)` | Applies the ReLU activation function. It is not currently used during training. |
