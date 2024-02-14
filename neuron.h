#ifndef NEURON_H
#define NEURON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<time.h>

typedef struct neuron_t {
    // Variable to store the value of this neuron's activation
    float actv;
   
    // This is a pointer to the array of floats that represent the weights connecting this neuron to the neurons in the next layer
    float *out_weights;
    
    // This is the bias term that is added onto the value of the weighted sum of the inputs
    float bias;
    
    // This variable will store the weighted sum of the inputs plus the bias, BEFORE applying the activation function
    float z;

    // This varibale will represent the derivative of this neuron's activation 
    float dactv;

    // This is a float pointer to an array of floats that represent the gradient of the weights ('outweights')
    // We track the gradient of the output weights since this is BACK propogation
    // We propogate the error from the output layers BACKWARDS to the input layers
    // For each neuron's output weights we track how much their value contributes to the loss function through this array of gradients
    float *dw;
    // This is the gradient of the bias term
	float dbias;
    // This is the gradient of the weighted sum of the inputs
	float dz;

    // TODO: Add function pointer for destructor

 } neuron;

neuron create_neuron(int num_out_weights);

#endif