typedef struct matrix {
    double **data;
    int rows;
    int columns;
}Matrix;


Matrix create_matrix(int rows, int columns) {
    Matrix matrix;

    int i, j;

    matrix.rows = rows;
    matrix.columns = columns;

    matrix.data = (double **)malloc(rows * sizeof(double*));
    for (i = 0; i < rows; i++) {
        matrix.data[i] = (double *)malloc(columns * sizeof(double));
        for (j = 0; j < columns; j++) {
            matrix.data[i][j] = 0.0;
        }
    }
    
    return matrix;
}

Matrix multiply_matrix(Matrix A, Matrix B) {
    Matrix C = create_matrix(A.rows, B.columns);
    
    int i, j, k;

    for (i = 0; i < C.rows; i++) {
        for (j = 0; j < C.columns; j++) {
            C.data[i][j] = 0.0;
            for (k = 0; k < B.rows; k++) {
                C.data[i][j] += A.data[i][k] * B.data[k][j];
            }
        }
    }

    return C;
}

void multiply_matrix_scalar(Matrix *A, double escalar) {
    int i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->columns; j++) {
            A->data[i][j] = A->data[i][j] * escalar;
        }
    }
}

Matrix transpose_matrix(Matrix A) {
    Matrix T = create_matrix(A.columns, A.rows);

    int i, j;
    for (i = 0; i < T.rows; i++) {
        for (j = 0; j < T.columns; j++) {
            T.data[i][j] = A.data[j][i]; 
        }
    }

    return T;
}

Matrix multiply_matrix_hadamard(Matrix A, Matrix B) {
    Matrix C = create_matrix(A.rows, A.columns);

    int i, j;
    for (i = 0; i < A.rows; i++) {
        for (j = 0; j < A.columns; j++) {
            C.data[i][j] = A.data[i][j] * B.data[i][j];
        }
    }

    return C;
}

Matrix sum_matrix(Matrix A, Matrix B) {
    Matrix C = create_matrix(A.rows, A.columns);

    int i, j;
    for (i = 0; i < A.rows; i++) {
        for (j = 0; j < A.columns; j++) {
            C.data[i][j] = A.data[i][j] + B.data[i][j];
        }
    }

    return C;
}

Matrix subtract_matrix(Matrix A, Matrix B) {
    Matrix C = create_matrix(A.rows, A.columns);

    int i, j;
    for (i = 0; i < A.rows; i++) {
        for (j = 0; j < A.columns; j++) {
            C.data[i][j] = A.data[i][j] - B.data[i][j];
        }
    }

    return C;
}

void randomize_matrix(Matrix *A) {
    int i, j;
    for (i = 0; i < A->rows; i++) {
        for (j = 0; j < A->columns; j++) {
            A->data[i][j] = ((double)rand()/(double)(RAND_MAX)) * 5.0;
        }
    }
}

Matrix array_2_matrix(double *inputs, int lenght) {
    Matrix matrix = create_matrix(lenght, 1);
    int i;
    for (i = 0; i < matrix.rows; i++) {
        matrix.data[i][0] = inputs[i];
    }
    return matrix;
}

void print_matrix(Matrix A) {
    int i, j;
    for (i = 0; i < A.rows; i++) {
        for (j = 0; j < A.columns; j++) {
            printf("%lf  ", A.data[i][j]);
        }
        printf("\n");
    }
}