#include "layer.h"

layer create_layer(int number_of_neurons) {
    layer layer;
    // We set the number of neurons in this layer to -1 to indicate that the actual neurons have not been created yet
    layer.num_neurons = -1;
    // We allocate to the neurons pointer the amount of meory needed for the number of neurons in this layer
    layer.neurons = (struct neuron_t *) malloc(number_of_neurons * sizeof(struct neuron_t));
    return layer;
}

// How to instance a lone layer in main:
    // TESTING layer instancing
    // We can now create a layer
    //int number_of_neurons = 2;
    //layer layer = create_layer(number_of_neurons);
    //printf("Number of neurons in layer: %d\n", layer.num_neurons);

// TODO:
// Add destructor