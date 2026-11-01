#include "layer_dims.h"

// first convolution layer turns 2 input channels into 16 output channels, each filter looks at 5 time samples, add 2 zeroes at the end for padding
// 2 -> 16, k = 5, pad = 2
void conv1_1d(const data_t inputs[C1_IC][C1_IN_LEN],
              data_t outputs[C1_OC][C1_OUT_LEN],
              const data_t weights[C1_OC][C1_IC][C1_K],
              const data_t bias[C1_OC]) {    
    
    for (int oc = 0; oc < C1_OC; ++oc) {
        #pragma HLS PIPELINE off // disable pipelining for oc loop
        for (int o = 0; o < C1_OUT_LEN; ++o) {
            acc_t acc = bias[oc]; //start w/ bias

            for (int ic = 0; ic < C1_IC; ++ic) {
                for (int k = 0; k < C1_K; ++k) {

                    // #pragma HLS PIPELINE II=1

                    int idx = o - C1_PAD + k; //input sample
                    data_t v = (idx >= 0 && idx < C1_IN_LEN) ? inputs[ic][idx] : (data_t)0; // use 0 if idx is outsde of input
                    acc += v * weights[oc][ic][k]; //10 weighted values
                }
            }
            outputs[oc][o] = (data_t)acc; //result
        }
    }
}