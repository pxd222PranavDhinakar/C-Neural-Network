# C Neural Network Framework

## Introduction

This project is a simple neural network framework implemented in C. It's designed to demonstrate the basics of neural network operations, including forward propagation, backpropagation, and training a network to learn patterns such as logical operations (e.g., XOR). This framework uses a small feedforward neural network architecture suitable for educational purposes and small-scale experimentation.

## Getting Started

### Prerequisites

Ensure you have a C compiler and `make` installed on your system. This project has been tested on Linux and macOS environments.

### Building the Project

The project includes a Makefile for easy compilation. To build the project, follow these steps:

1. Open a terminal.
2. Navigate to the project's root directory.
3. Run the following command:

   ```sh
   make
   ```

This command will compile the project and create an executable named `project` in the project directory.

### Running the Project

After building the project, you can run it by executing the following command in the project's root directory:

```sh
./project
```

This will execute the neural network framework and perform the operations defined in the main function, such as training the network and testing it with predefined inputs.

### Model Architecture
Currently you can define the number of layers and the number of neurons in each layer in the main.c file. The model is currently configured to learn simple logic gates. Right now it is learning the AND gate.

As a result the model's architecture is currently very simple and is as follows:
- Input layer: 2 neurons
- Hidden layer: 4 neurons
- Output layer: 1 neuron

The model uses a ReLU activation function for the hidden layer.

The model uses a Sigmoid activation function for the output layer. 

The learning rate is set to 0.05 and the number of training epochs is set to 2000.

### Understanding Output of ./project
The model is currently configured to learn simple logic gates. Right now it is learning the XOR gate.

If you wish to test learning the AND gate just comment the section that specifies the data for XOR and uncomment the section that specifies the data for AND. These sections should be near the top of the main.c file, after the includes and function prototypes.

The output for learning the AND gate should look like this:
```
Testing the trained neural network:
Output: 0
Inputs: 0.0, 0.0 | Expected: 0.0 | Predicted: 0.00

Output: 0
Inputs: 0.0, 1.0 | Expected: 1.0 | Predicted: 0.04

Output: 0
Inputs: 1.0, 0.0 | Expected: 1.0 | Predicted: 0.04

Output: 1
Inputs: 1.0, 1.0 | Expected: 0.0 | Predicted: 0.94
```

The output for learning the XOR gate should look like this:
```
Testing the trained neural network:
Output: 0
Inputs: 0.0, 0.0 | Expected: 0.0 | Predicted: 0.07

Output: 1
Inputs: 0.0, 1.0 | Expected: 1.0 | Predicted: 0.91

Output: 1
Inputs: 1.0, 0.0 | Expected: 1.0 | Predicted: 0.95

Output: 0
Inputs: 1.0, 1.0 | Expected: 0.0 | Predicted: 0.06
```
## Dependencies

This project is standalone and does not require external libraries. It only depends on the standard C library for its functionality.

## Usage

The main functionality of this neural network framework is encapsulated in the provided C files. Users can modify the `main.c` file or create new modules to define different neural network architectures, learning problems, or experiments.

For detailed usage examples, please refer to the source code comments or documentation provided within the codebase.

## Contributing

Contributions to improve the framework or extend its capabilities are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is open-sourced under the MIT license. See the LICENSE file for details.
