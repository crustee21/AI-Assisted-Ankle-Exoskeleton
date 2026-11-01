#include "layer_dims.h"

//ReLU and Max pooling to make nonlinear and smaller
void relu_max_1(const data_t input[P1_C][P1_IN_LEN],
                data_t output[P1_C][P1_OUT_LEN]) {

    CH: for (int r = 0; r < P1_C; r++) {
        POOL: for (int o = 0; o < P1_OUT_LEN; o++) {
            data_t a = input[r][2*o];
            data_t b = input[r][2*o + 1];
            data_t m = (a > b) ? a : b;                     // max before relu
            output[r][o] = (m > (data_t)0) ? m : (data_t)0;
        }
    }
}