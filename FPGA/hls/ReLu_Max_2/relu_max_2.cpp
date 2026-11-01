#include "layer_dims.h"

//ReLU and Max pooling to make nonlinear and smaller
void relu_max_2(const data_t input[P2_C][P2_IN_LEN],
                data_t output[P2_C][P2_OUT_LEN]) {

    CH: for (int r = 0; r < P2_C; r++) {
        POOL: for (int o = 0; o < P2_OUT_LEN; o++) {
            data_t a = input[r][2*o];
            data_t b = input[r][2*o + 1];
            data_t m = (a > b) ? a : b;
            output[r][o] = (m > (data_t)0) ? m : (data_t)0;
        }
    }
}