#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include "matrix.h"

#define get_lenght(x) (sizeof(x)/sizeof(x[0]))

typedef struct layer {
    Matrix layer_data;
    Matrix weights;
}Layer;


typedef struct nn {
    Layer input;
    Layer *hidden;
    Layer output;
    int h_layers;
    double learn_rate;
}NeuralNetwork;

    void print_neuralnetwork(NeuralNetwork *nn);
    NeuralNetwork *create_NN(int i_nodes, int h_nodes, int h_layers, int o_nodes);
    void set_input_layer(NeuralNetwork *nn, double *arr);
    void feedforward(NeuralNetwork *nn);
    double sigmoid(double x);
    double dsigmoid(double x);
    void activation(Matrix *matrix);
    void backpropagation(NeuralNetwork *nn, double *target);
    Matrix derivate(Matrix A);

int main() {

    srand(time(0));

    double input_array[] = {1, 2, 3, 4};
    double target_array[] = {3, 7};

    NeuralNetwork *nn = create_NN(5, 4, 3, 2);
    
    set_input_layer(nn, input_array);

    //print_neuralnetwork(nn);
    
    feedforward(nn);

    backpropagation(nn, target_array);

    feedforward(nn);

    return 0;
}

void backpropagation(NeuralNetwork *nn, double *target) {
    int last = nn->h_layers-1;

    // dW = erro (*) d(output) * lr * Hidden_T
    Matrix target_matrix = array_2_matrix(target, nn->output.layer_data.rows);
    
    // OUTPUT -> LAST_HIDDEN
    Matrix error = subtract_matrix(target_matrix, nn->output.layer_data);
    Matrix d_output = derivate(nn->output.layer_data);
    Matrix last_hidden_T = transpose_matrix(nn->hidden[last].layer_data);
    Matrix gradient = multiply_matrix_hadamard(error, d_output);
    multiply_matrix_scalar(&gradient, nn->learn_rate);
    Matrix dW = multiply_matrix(gradient, last_hidden_T);
    nn->hidden[last].weights = sum_matrix(nn->hidden[last].weights, dW);
    
    printf("\n[%d]\n", last);
    print_matrix(nn->hidden[last].weights);

    int i;
    for (i = last - 1; i >= 0; i--) {
        printf("\n[%d]\n", i);
        print_matrix(nn->hidden[i].weights);

        Matrix weights_T = transpose_matrix(nn->hidden[i].weights);
        error = multiply_matrix(weights_T, error);
        Matrix d_hidden = derivate(nn->hidden[i].layer_data);
        Matrix back_hidden_T = transpose_matrix(nn->hidden[i].layer_data);
        Matrix gradient_H = multiply_matrix_hadamard(error, d_hidden);
        multiply_matrix_scalar(&gradient_H, nn->learn_rate);
        Matrix dWH = multiply_matrix(gradient_H, back_hidden_T);

        nn->hidden[i].weights = sum_matrix(nn->hidden[i].weights, dWH);

        printf("\n");
        print_matrix(nn->hidden[i].weights);
    }

    i++;

    printf("\n[INPUTS]\n");
    print_matrix(nn->input.weights);

    Matrix weights_T = transpose_matrix(nn->hidden[i].weights);
    error = multiply_matrix(weights_T, error);
    Matrix d_hidden = derivate(nn->hidden[i].layer_data);
    Matrix input_T = transpose_matrix(nn->input.layer_data);
    Matrix gradient_H = multiply_matrix_hadamard(error, d_hidden);
    multiply_matrix_scalar(&gradient_H, nn->learn_rate);
    Matrix dWH = multiply_matrix(gradient_H, input_T);
    nn->input.weights = sum_matrix(nn->input.weights, dWH);

    printf("\n");
    print_matrix(nn->input.weights);
}

void feedforward(NeuralNetwork *nn) {
    int i;

    printf("\ninput\n");
    print_matrix(nn->input.layer_data);

    // INPUT -> HIDDEN[0]
    nn->hidden[0].layer_data = multiply_matrix(nn->input.weights, nn->input.layer_data);
    activation(&nn->hidden[0].layer_data);

    printf("\nhidden_output[0]\n");
    print_matrix(nn->hidden[0].layer_data);

    // HIDDEN[i] -> HIDDEN[i - 1]
    for (i = 1; i < nn->h_layers; i++) {
        nn->hidden[i].layer_data = multiply_matrix(nn->hidden[i - 1].weights, nn->hidden[i - 1].layer_data);
        activation(&nn->hidden[i].layer_data);
        printf("\nhidden_output[%d]\n", i);
        print_matrix(nn->hidden[i].layer_data);
    }

    // LAST_HIDDEN -> OUTPUT
    nn->output.layer_data = multiply_matrix(nn->hidden[i - 1].weights, nn->hidden[i - 1].layer_data);
    activation(&nn->output.layer_data);
    printf("\noutput\n");
    print_matrix(nn->output.layer_data);

}

void activation(Matrix *matrix) {
    int i, j;
    for (i = 0; i < matrix->rows; i++) {
        for (j = 0; j < matrix->columns; j++) {
            matrix->data[i][j] = sigmoid(matrix->data[i][j]);
        }
    }
}

void set_input_layer(NeuralNetwork *nn, double *arr) {
    int i, j;
    for (i = 0; i < nn->input.layer_data.rows; i++) {
        for (j = 0; j < nn->input.layer_data.columns; j++) {
            nn->input.layer_data.data[i][j] = arr[i];
        }
    }
}

NeuralNetwork *create_NN(int h_layers, int i_nodes, int h_nodes, int o_nodes) {
    NeuralNetwork *nn = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    
    nn->learn_rate = 0.1;

    nn->h_layers = h_layers;

    nn->input.layer_data = create_matrix(i_nodes, 1);
    nn->input.weights = create_matrix(h_nodes, i_nodes);
    randomize_matrix(&nn->input.weights);
    
    nn->hidden = (Layer *)malloc(h_layers * sizeof(Layer));

    int i;
    for (i = 0; i < h_layers; i++) {
        if (i == h_layers-1) {
            nn->hidden[i].weights = create_matrix(o_nodes, h_nodes);
        } else {
            nn->hidden[i].weights = create_matrix(h_nodes, h_nodes);
        }
        randomize_matrix(&nn->hidden[i].weights);
    }

    return nn;
}

void print_neuralnetwork(NeuralNetwork *nn) {
    printf("Input Layer:\n");
    print_matrix(nn->input.layer_data);
    
    int i;
    for (i = 0; i < nn->h_layers; i++) {
        printf("\nHidden Layer[%d] weights\n", i);
        print_matrix(nn->hidden[i].weights);
    }
}

double sigmoid(double x) {
     double exp_value;
     double return_value;

     exp_value = exp((double) -x);

     return_value = 1 / (1 + exp_value);

     return return_value;
}

double dsigmoid(double x) {
    return x * (1 - x);
}

Matrix derivate(Matrix A) {
    Matrix D = create_matrix(A.rows, A.columns);
    
    int i, j;
    for (i = 0; i < A.rows; i++) {
        for (j = 0; j < A.columns; j++) {
            D.data[i][j] = dsigmoid(A.data[i][j]); 
        }
    }
    
    return D;
}