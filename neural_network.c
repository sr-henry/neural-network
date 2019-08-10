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

    double input_array[4][2] = {{1, 1}, {1, 0}, {0, 1}, {0, 0}};
    double target_array[4][1] = {{0}, {1}, {1}, {0}};

    NeuralNetwork *nn = create_NN(2, 2, 4, 1);

    int i, index;
    for (i = 0; i < 100000; i++) {
        index = rand()%4;

        printf("%lf : %lf\t%lf\t", input_array[index][0], input_array[index][1], target_array[index][0]);

        set_input_layer(nn, input_array[index]);
        feedforward(nn);
        backpropagation(nn, target_array[index]);

        print_matrix(nn->output.layer_data);
        
    }
    
    return 0;
}

void backpropagation(NeuralNetwork *nn, double *target) {

    int index = nn->h_layers - 1;

    // dW = erro (*) d(output_right) * lr * left_layer_T
    Matrix target_output = array_2_matrix(target, nn->output.layer_data.rows);

    // Calculate OutputError
    Matrix output_errors = subtract_matrix(target_output, nn->output.layer_data);

    /*OUTPUT_LAYER -> LAST_HIDDEN */
    Matrix d_outputs = derivate(nn->output.layer_data);
    Matrix hidden_T = transpose_matrix(nn->hidden[index].layer_data);
    Matrix gradient = multiply_matrix_hadamard(output_errors, d_outputs);
    multiply_matrix_scalar(&gradient, nn->learn_rate);
    Matrix delta_w = multiply_matrix(gradient, hidden_T);
    nn->hidden[index].weights = sum_matrix(nn->hidden[index].weights, delta_w);
    
    /*LAST_HIDDEN -> FIRST_HIDDEN */
    int i;
    for (i = (index - 1); i >= 0; i--) {
        Matrix weights_T = transpose_matrix(nn->hidden[i].weights);
        Matrix hidden_error = multiply_matrix(weights_T, output_errors);
        d_outputs = derivate(nn->hidden[i + 1].layer_data);
        hidden_T = transpose_matrix(nn->hidden[i].layer_data);
        gradient = multiply_matrix_hadamard(hidden_error, d_outputs);
        multiply_matrix_scalar(&gradient, nn->learn_rate);
        delta_w = multiply_matrix(gradient, hidden_T);
        nn->hidden[i].weights = sum_matrix(nn->hidden[i].weights, delta_w);
    }

    i++;

    /*FIRST_HIDDEN -> INPUT_LAYER */
    Matrix weights_T = transpose_matrix(nn->hidden[i].weights);
    Matrix input_error = multiply_matrix(weights_T, output_errors);
    d_outputs = derivate(nn->hidden[i].layer_data);
    Matrix inputs_T = transpose_matrix(nn->input.layer_data);
    gradient = multiply_matrix_hadamard(input_error, d_outputs);
    multiply_matrix_scalar(&gradient, nn->learn_rate);
    delta_w = multiply_matrix(gradient, inputs_T);
    nn->input.weights = sum_matrix(nn->input.weights, delta_w);

}

void feedforward(NeuralNetwork *nn) {
    int i;

    //printf("\ninput\n");
    //print_matrix(nn->input.layer_data);

    // INPUT -> HIDDEN[0]
    nn->hidden[0].layer_data = multiply_matrix(nn->input.weights, nn->input.layer_data);
    activation(&nn->hidden[0].layer_data);

    //printf("\nhidden_output[0]\n");
    //print_matrix(nn->hidden[0].layer_data);

    // HIDDEN[i] -> HIDDEN[i - 1]
    for (i = 1; i < nn->h_layers; i++) {
        nn->hidden[i].layer_data = multiply_matrix(nn->hidden[i - 1].weights, nn->hidden[i - 1].layer_data);
        activation(&nn->hidden[i].layer_data);
        //printf("\nhidden_output[%d]\n", i);
        //print_matrix(nn->hidden[i].layer_data);
    }

    // LAST_HIDDEN -> OUTPUT
    nn->output.layer_data = multiply_matrix(nn->hidden[i - 1].weights, nn->hidden[i - 1].layer_data);
    activation(&nn->output.layer_data);
    //printf("\noutput\n");

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