#include "neuron.h"

// Function to create and return a neuron
// This function takes in the number of output weights that this neuron will have
neuron create_neuron(int num_out_weights) {
    neuron neuron;
    // Initialize activation as 0
    neuron.actv = 0.0;

    // Allocate in memory the amount of space needed for the number of output weights
    // Use (float *) to create a pointer to the array of floats
    neuron.out_weights = (float *) malloc(num_out_weights * sizeof(float));
    // Only if you want all weights to be initially zero
    //memset(neuron.out_weights, 0, num_out_weights * sizeof(float));
    // Print out the all of the addresses of all the output weights
    //printf("Addresses of the output weights IN CONSTRUCTOR: \n");
    //for(int i = 0; i < num_out_weights; i++) {
    //    printf("%p\n", &neuron.out_weights[i]);
    //}

    // Initialize bias as 0
    neuron.bias = 0.0;

    // Initialize weighed sum value as 0
    neuron.z = 0.0;

    // Initialize the derivative of the activation as 0
    neuron.dactv = 0.0;

    // Allocate in memory the amount of space needed for the number of output weight gradients
    neuron.dw = (float *) malloc(num_out_weights * sizeof(float));
    // Initialize the gradient of the bias as 0
    neuron.dbias = 0.0;
    // Iitialize the gradient of the weighted sum as 0
    neuron.dz = 0.0;

    // COMMAND THAT OUTPUTS WHETHER ALL ELEMENTS OF THIS NEURON HAVE BEEN INITIALIZED
    printf("Neuron Initialized with num outputs: %d\n", num_out_weights);

    // Finally return the fully initialized neuron
    return neuron;
}


// How to instance a lone neuron in main:
    // TESTING neuron instancing
    // We can now create a neuron
    //int num_out_weights = 2;
    //neuron neuron = create_neuron(num_out_weights);

    //printf("Value of neuron activation: %f\n", neuron.actv);


// TODO:
// Add destructor