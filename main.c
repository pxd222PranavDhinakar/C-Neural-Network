#include "backprop.h"
#include "layer.h"
#include "neuron.h"


// Pointer to the layers array of the neural network. This will hold all the layers of the neural network, 
// including the input layer, hidden layers, and output layer. Each layer can contain multiple neurons.
layer *layers = NULL;

// Integer to store the number of layers in the neural network. This includes all types of layers 
// (input, hidden, and output).
int num_layers;

// Pointer to the array of integers that represents the number of neurons in each layer.
// This array's length will be equal to 'num_layers', and each element stores the count of neurons in the corresponding layer.
int *num_neurons;

// The learning rate of the neural network. It controls how much we adjust the weights of our network 
// with respect the loss gradient. Lower values mean slower learning.
float alpha;

// Pointer to an array that will store the cost (error) for each training example. This is used to calculate 
// the gradient during backpropagation.
float *cost;

// A single float to store the total cost (error) over all training examples. This is useful for evaluating 
// the overall performance of the network after an iteration of training.
float full_cost;

// A pointer to a pointer representing a 2D array for input data. Each row corresponds to one training example,
// and each column represents a feature of the input data.
float **input;

// A pointer to a pointer representing a 2D array for the desired (target) output data for each training example.
// This is used to compute the cost (error) during training.
float **desired_outputs;

// Integer to store the number of training examples. This is used to iterate over the 'input' and 
// 'desired_output' arrays during training.
int num_training_examples;

// An example counter or parameter, initially set to 1. Its specific purpose will depend on its use 
// within the neural network's learning or operation phases.
int n = 1;


// STATUS CODES
// A status code of 0 indicates success, while a status code of -1 indicates an error.
// Status code indicating successful creation of the neural network architecture.
#define SUCCESS_CREATE_ARCHITECTURE 0
// Status code indicating an initialization error occurred.
#define ERR_INIT -1
// Status code indicating successful initialization of the neural network.
#define SUCCESS_INIT 0
// Status code indicating successful initialization of the weights.
#define SUCCESS_INIT_WEIGHTS 0
// Status code indicating an error occurred during the initialization of the weights.
#define ERR_INIT_WEIGHTS -1
// Status code indicating an error occurred during the creation of the neural network architecture.
#define ERR_CREATE_ARCHITECTURE -1
// Status code indicating successful destruction of the neural network.
#define SUCCESS_DINIT 0


// Function prototypes
int init(void);
int create_architecture(void);
int initialize_weights(void);
void get_inputs(void);
void print_dataset(void);
void train_neural_net(void);
void feed_input(int i);
void forward_prop(void);
void compute_cost(int i);
void back_prop(int p);
void update_weights(void);
void test_nn(void);
int dinit(void);


/*
// LEARNING XOR GATE
// Funciton to define training data
void get_inputs(void) {
    // Define the XOR inputs
    input[0][0] = 0; input[0][1] = 0;
    input[1][0] = 0; input[1][1] = 1;
    input[2][0] = 1; input[2][1] = 0;
    input[3][0] = 1; input[3][1] = 1;
}

// Function to define the label data
void get_desired_outputs(void) {
    // Define the XOR desired outputs
    desired_outputs[0][0] = 0;
    desired_outputs[1][0] = 1;
    desired_outputs[2][0] = 1;
    desired_outputs[3][0] = 0;
}
*/

// LEARNING AND GATE
// Function to define training data for the AND logic gate
void get_inputs(void) {
    input[0][0] = 0; input[0][1] = 0;
    input[1][0] = 0; input[1][1] = 1;
    input[2][0] = 1; input[2][1] = 0;
    input[3][0] = 1; input[3][1] = 1;
}

// Function to define the label data for the AND logic gate
void get_desired_outputs(void) {
    desired_outputs[0][0] = 0;
    desired_outputs[1][0] = 0;
    desired_outputs[2][0] = 0;
    desired_outputs[3][0] = 1;
}


void print_dataset() {
    int i, j;

    printf("Training dataset (Inputs and Desired Outputs):\n");
    for (i = 0; i < num_training_examples; i++) {
        // Print inputs
        printf("Input[%d]: ", i);
        for (j = 0; j < num_neurons[0]; j++) { // Assuming num_neurons[0] is the number of input features
            printf("%.1f ", input[i][j]);
        }

        // Print corresponding desired output
        printf(" | Desired Output: ");
        for (j = 0; j < num_neurons[num_layers - 1]; j++) { // Assuming num_neurons[num_layers - 1] is the number of outputs
            printf("%.1f", desired_outputs[i][j]);
        }
        printf("\n");
    }
}


int init() {
    // If the architecture is not created successfully, return an error
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE) {
        printf("Error in creating architecture...\n");
        return ERR_INIT;
    }

    // If the architecture is created successfully, print a success message
    printf("Neural Network Created Successfully...\n\n");
    return SUCCESS_INIT;
}

int initialize_weights(void) {
    int i,j,k;

    // Check if any layers exist
    if(layers == NULL) {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initializing Weights...\n");
    
    // Loop through the layers till the second last layer
    for(i = 0; i < num_layers-1; i ++) {
                
        // Loop through the neurons in the ith layer
        for(j = 0; j < num_neurons[i]; j++) {
            
            // Print the current neuron number and layer number
            //printf("Neuron: %d in Layer: %d\n", j, i);

            // Loop through the number of outputs weights the jth neuron will need to have
            for(k = 0; k < num_neurons[i+1]; k++) {

                // Now let us test accessing the weights of the neurons in the layers
                //float *weight_k = &layers[i].neurons[j].out_weights[k];
                //printf("IN FUNCTION ADDRESS: %p\n", weight_k);

                // Assign a value to this weight
                //*weight_k = 6.66; // We dereference the pointer to assign a value to the actual weight variable
                //printf("Value of weight %d of Neuron %d in Layer %d: %f\n", k, j, i, layers[i].neurons[j].out_weights[k]);


                // Set the kth output weight of the jth neuron in the ith layer to a random value between 0 and 1
                layers[i].neurons[j].out_weights[k] = ((double)rand())/((double)RAND_MAX);
                printf("%d:w[%d][%d]: %f\n",k,i,j, layers[i].neurons[j].out_weights[k]);
                // Initialize the gradient of the kth output weight of the jth neuron in the ith layer to 0
                layers[i].neurons[j].dw[k] = 0.0;
                //printf("%d:dw[%d][%d]: %f\n",k,i,j, layers[i].neurons[j].dw[k]);
            }

            // If we are passed the input layer, we need to initialize bias term to a random value between 0 and 1
            if(i > 0) {
                layers[i].neurons[j].bias = ((double)rand())/((double)RAND_MAX);
            }
        }
    }
    printf("\n");

    // Loop through the number of neurons in the last layer
    for (j=0; j<num_neurons[num_layers-1]; j++) {
        // Initialize the bias term of the jth neuron in the last layer to a random value between 0 and 1
        layers[num_layers-1].neurons[j].bias = ((double)rand())/((double)RAND_MAX);
    }

    return SUCCESS_INIT_WEIGHTS;
}



int create_architecture() {
    // Local iterator variables
    int i = 0;
    int j = 0;
    // Asign to the global variable layers the memory allocation for the number of layers
    layers = (layer *) malloc(num_layers * sizeof(layer)); // Allocate memory for the layers array

    // Loop through layers 
    for(i = 0; i < num_layers; i++) {

        layers[i] = create_layer(num_neurons[i]); // Create a layer with the number of neurons in the ith layer
        layers[i].num_neurons = num_neurons[i]; // Set the number of neurons in the ith layer
        printf("Created Layer: %d\n", i);
        printf("Number of Neurons in Layer %d: %d\n", i, layers[i].num_neurons);

        // Loop through each neuron in the ith layer
        for(j = 0; j < num_neurons[i]; j++) {
            // If we are not at the last layer, we need to populate this layer with neurons
            if(i < num_layers - 1) { // MADE A CHANGE HERE FIX LATER
                // Initialize the jth neuron in the ith layer
                // Specify that this neuron has num_neurons[i+1] output weights as neurons[i+i] represents the number of neurons in the next layer
                layers[i].neurons[j] = create_neuron(num_neurons[i+1]);

                //printf("Address of Neuron %d in Layer %d: %p\n", j, i, &layers[i].neurons[j]);
                // Printing the addresses of the weight floats
            }

            printf("Neuron %d in Layer %d created...\n", j, i);
        }
        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if(initialize_weights() != SUCCESS_INIT_WEIGHTS) {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

// Feeds inputs to the input layer
void feed_input(int i) {
    int j;

    // Loop through the neurons in the input layer
    for(j = 0; j < num_neurons[0]; j++) {
        // Set the activation of the jth neuron in the input layer to the jth feature of the ith training example
        layers[0].neurons[j].actv = input[i][j];
        printf("Input[%d][%d]: %f\n", i, j, layers[0].neurons[j].actv);
    }
}

// Forward Propagation
// This function will be used to feed the input data to the input layer, and then propagate the data through the network
// to compute the output of the network
void forward_prop(void) {
    int i,j,k;
    
    // Start looping from the first hidden layer
    // As we have already fed the input data to the input layer
    for(i = 1; i < num_layers; i++) {
        // Loop through all the neurons in the current layer
        for(j = 0; j < num_neurons[i]; j++) {
            // Add the initial bias of the neuron to the weighted sum of the inputs
            layers[i].neurons[j].z = layers[i].neurons[j].bias;

            // Now loop through the outputs of the previous layer
            for(k = 0; k < num_neurons[i - 1]; k++) {
                // Add on the weighted sum of the inputs
                // We multiply the value of the current weight to the activation of the previous neuron
                layers[i].neurons[j].z = layers[i].neurons[j].z + ((layers[i-1].neurons[k].out_weights[j])* (layers[i-1].neurons[k].actv));
            }

            // ReLU Activation Function for Hidden Layers
            if(i < num_layers-1){
                // If the weighted sum of the inputs is less than 0, set the activation of the neuron to 0
                if((layers[i].neurons[j].z) < 0){
                    layers[i].neurons[j].actv = 0;
                }
                else{
                    layers[i].neurons[j].actv = layers[i].neurons[j].z;
                }
            }

            // Sigmoid Activation Function for Output Layer
            else{
                layers[i].neurons[j].actv = 1/(1+exp(-layers[i].neurons[j].z));
                printf("Output: %d\n", (int)round(layers[i].neurons[j].actv));
                //printf("\n");
            }
        }
    }
}

// Compute the loss for a single training example
// This is an MSE (Mean Squared Error) loss function
void compute_cost(int i) {
    int j;
    float tmpcost = 0.0;
    float tcost = 0;

    // Loop through the neurons in the output layer
    for(j = 0; j < num_neurons[num_layers-1]; j ++) {
        // Calculate the cost for the jth output neuron
        tmpcost = desired_outputs[i][j] - layers[num_layers-1].neurons[j].actv;
        cost[j] = (tmpcost * tmpcost)/2;
        tcost = tcost + cost[j];
    }
    
    full_cost = (full_cost + tcost)/n;
    n++;

    printf("Full Cost: %f\n",full_cost);
}


// Backward Propagation
// This function will be used to compute the gradients of the weights and biases of the network
// This is done by computing the gradients of the cost function with respect to the weights and biases
// This is done using the chain rule of calculus
void back_prop(int p) {
    int i,j,k;

    // Loop through the neurons in the output layer
    for(j = 0; j < num_neurons[num_layers-1]; j++) {
        // Compute the derivative of the activation of the jth neuron in the output layer
        layers[num_layers-1].neurons[j].dz = (layers[num_layers-1].neurons[j].actv - desired_outputs[p][j]) * (layers[num_layers-1].neurons[j].actv) * (1- layers[num_layers-1].neurons[j].actv);

        // Loop through the neurons in the previous layer
        for(k = 0; k < num_neurons[num_layers-2]; k++) {
            // Compute the gradient of the kth output weight of the jth neuron in the output layer
            layers[num_layers-2].neurons[k].dw[j] = (layers[num_layers-1].neurons[j].dz * layers[num_layers-2].neurons[k].actv);

            // Compute the gradient of the activation of the kth neuron in the previous layer
            layers[num_layers-2].neurons[k].dactv = layers[num_layers-2].neurons[k].out_weights[j] * layers[num_layers-1].neurons[j].dz;
        }
        // Compute the gradient of the bias of the jth neuron in the output layer
        layers[num_layers-1].neurons[j].dbias = layers[num_layers-1].neurons[j].dz;
    }

    // Loop backward through the hidden layers
    for(i = num_layers - 2; i > 0; i--) {
        // Loop through the neurons in the current layer
        for(j = 0; j < num_neurons[i]; j++) {
            // If the activation of the jth neuron in the ith layer is greater than 0
            if(layers[i].neurons[j].z >= 0) {
                // Set the derivative of the weighted sum of the inputs to the derivative of the activation of the jth neuron in the ith layer
                layers[i].neurons[j].dz = layers[i].neurons[j].dactv;
            }
            else {
                // Set the derivative of the weighted sum of the inputs to 0
                layers[i].neurons[j].dz = 0;
            }

            // Loop through the neurons in the previous layer
            for(k = 0; k < num_neurons[i - 1]; k++) {
                // Compute the gradient of the kth output weight of the jth neuron in the ith layer
                layers[i-1].neurons[k].dw[j] = layers[i].neurons[j].dz * layers[i-1].neurons[k].actv;    
                
                // If we are not at the input layer
                if(i>1) {
                    // Set the gradient of the activation of the kth neuron in the previous layer
                    layers[i-1].neurons[k].dactv = layers[i-1].neurons[k].out_weights[j] * layers[i].neurons[j].dz;
                }
            }

            // Compute the gradient of the bias of the jth neuron in the ith layer
            layers[i].neurons[j].dbias = layers[i].neurons[j].dz;
        }
    }

}

// Function to update the weights and biases of the network
void update_weights(void) {
    int i,j,k;

    // Lopp through the layers up until the output layer
    for(i = 0; i < num_layers-1; i++) {
        // Loop through the neurons in the current layer
        for(j = 0; j < num_neurons[i]; j++) {
            // Loop through the output weights of the jth neuron in the ith layer
            for(k = 0; k < num_neurons[i+1]; k++) {
                // Update the kth output weight of the jth neuron in the ith layer
                layers[i].neurons[j].out_weights[k] = (layers[i].neurons[j].out_weights[k]) - (alpha * layers[i].neurons[j].dw[k]);
            }
            
            // Update Bias
            layers[i].neurons[j].bias = layers[i].neurons[j].bias - (alpha * layers[i].neurons[j].dbias);
        }
    }
}

void train_neural_net(void) {
    int i;
    int it = 0;

    // Gradient Descent
    // Iterations of training
    for(it = 0; it < 2000; it++) {
        for(i = 0; i < num_training_examples; i++) {
            // Feed input
            feed_input(i);

            // Forward Propagation
            forward_prop();

            // Compute cost
            compute_cost(i);

            // Backward Propagation
            back_prop(i);

            // Update weights
            update_weights();
        }
    }
}



// Test the trained network
/*
void test_nn(void) {
    int i;
    while(1)
    {
        printf("Enter input to test:\n");

        for(i=0;i<num_neurons[0];i++)
        {
            scanf("%f",&layers[0].neurons[i].actv);
        }
        forward_prop();
    }
}
*/


#include <math.h> // For fabs function

// Test the neural network's ability to predict XOR outputs
void test_nn(void) {
    int i, j;
    float prediction;

    // Define the test dataset (same as the training dataset for XOR)
    float test_inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    float expected_outputs[4] = {0, 1, 1, 0};

    printf("\n");
    printf("Testing the trained neural network:\n");
    for(i = 0; i < 4; i++) {
        // Set the inputs to the network
        for(j = 0; j < num_neurons[0]; j++) {
            layers[0].neurons[j].actv = test_inputs[i][j];
        }

        // Perform forward propagation to predict output
        forward_prop();

        // The prediction is the output of the last layer's first neuron
        prediction = layers[num_layers - 1].neurons[0].actv;
        
        // Print the results
        printf("Inputs: %.1f, %.1f | Expected: %.1f | Predicted: %.2f\n", 
               test_inputs[i][0], test_inputs[i][1], 
               expected_outputs[i], prediction);
        printf("\n");
    }
}



// TODO: Add different Activation functions
//void activation_functions()

/*
int dinit(void) {
    // TODO:
    // Free up all the structures

    return SUCCESS_DINIT;
}
*/



int main (void) {
    int i;

    // Define the number of layers in the neural network
    num_layers = 3; // This defines a 3 layer neural network 1 input, 1 hidden, 1 output
    // Allocate memory for a single neuron in each layer
    num_neurons = (int *) malloc(num_layers * sizeof(int)); // We are allocating memory for 3 integers
    
    // We then set the number of nurons in each layer
    // Here memset is a function that sets the first num_layers * sizeof(int) bytes of the block of memory pointed by num_neurons to the specified value
    // This function sets each byte in the allocated memory block to a value of 0
    // The first argument is the pointer to the memory block
    // The second argument is the value to be set
    // The third argument is the number of bytes in the block to be set to the value, here we are passing in the exact size of the block
    memset(num_neurons, 0, num_layers * sizeof(int)); // Each integer value in the block is initialized to 0, later each ith element will be set to the number of neurons in the ith layer

    // Now let's define the number of neurons in each layer
    int num_neurons_in_layer[] = {2, 3, 1}; // This is an array of integers that represents the number of neurons in each layer
    for(i = 0; i < num_layers; i++) {
        num_neurons[i] = num_neurons_in_layer[i]; // We set the number of neurons in each layer to the values in the array
    }
    // Now we have defined the structure of each layer in the neural network. 
    // Our model has 2 inputs, 3 neurons in the hidden layer, and 1 output neuron

    // Now let us initialize the neural network
    if (init() != SUCCESS_INIT) {
        printf("Error in Initialization...\n");
        exit(0);
    }

    // Define the learning rate for the model
    alpha = 0.05;

    // Define the number of training examples
    num_training_examples = 4;

    // Allocate memory for the input data
    // Defines a set of 4 training examples, each with 2 features
    input = (float **) malloc(num_training_examples * sizeof(float *));
    for(i = 0; i < num_training_examples; i++) {
        // Defines fore each training example, a set of 2 features
        // As the model has two inputs
        input[i] = (float *)malloc(num_neurons[0] * sizeof(float));  
    }

    desired_outputs = (float **) malloc(num_training_examples * sizeof(float *));
    for(i = 0; i < num_training_examples; i++) {
        // Defines fore each training example, a set of 1 output: LABEL
        // As the model has one output
        desired_outputs[i] = (float *)malloc(num_neurons[num_layers-1] * sizeof(float));
    }

    // Allocate memory for the value of the cost for each training output
    // Each training example will have a single output cost as this model has a single output
    cost = (float *) malloc(num_neurons[num_layers-1] * sizeof(float));
    // Set the initial value of this cost to 0
    memset(cost,0,num_neurons[num_layers-1]*sizeof(float));

    // Now that we have allocated memory for the input and output data, we can define the training data
    // Define Inputs
    get_inputs();

    // Define the desired outputs
    get_desired_outputs();

    // Print the dataset
    print_dataset();

    // Now we train the neural network
    train_neural_net();

    // Test the trained network
    test_nn();

    //if(dinit()!= SUCCESS_DINIT) {
    //    printf("Error in Dinitialization...\n");
    //}

    return 0;
}