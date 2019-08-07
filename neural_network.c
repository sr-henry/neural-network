#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include "matrix.h"

#define get_lenght(x) (sizeof(x)/sizeof(x[0]))

typedef struct nn {
    Matrix inputs;
    Matrix weights_ih;
    Matrix hidden;
    Matrix weights_ho;
    Matrix output;
    double learn_rate;
}NeuralNetwork;

    NeuralNetwork create_NN(int i_nodes, int h_nodes, int o_nodes);

int main()
{
    int i_nodes = 2;
    int h_nodes = 3;
    int o_nodes = 2;

    NeuralNetwork nn = create_NN(i_nodes, h_nodes, o_nodes);

    puts("NeuralNetwork Structure");
    printf("input: %d\nhidden: %d\noutput: %d\n\n", i_nodes, h_nodes, o_nodes);

    puts("weights [input_layer -> hidden_layer]");
    print_matrix(nn.weights_ih);

    puts("\nweights [hidden_layer -> output_layer]");
    print_matrix(nn.weights_ho);

    return 0;
}

NeuralNetwork create_NN(int i_nodes, int h_nodes, int o_nodes) {
    NeuralNetwork nn;

    nn.learn_rate = 0.1;

    nn.weights_ih = create_matrix(h_nodes, i_nodes);
    randomize_matrix(&nn.weights_ih);

    nn.weights_ho = create_matrix(o_nodes, h_nodes);
    randomize_matrix(&nn.weights_ho);

    return nn;
}