#ifndef LAYER_H
#define LAYER_H

#include "neuron.h"

// Now that we have a neuron struct, we can define a layer struct for the neuron's to reside within
typedef struct layer_t{
    // This is the number of neuron structs in this layer
    int num_neurons;

    // This is a pointer to an array of neuron structs in this layer
    struct neuron_t *neurons;
} layer;

layer create_layer(int num_neurons);


#endif