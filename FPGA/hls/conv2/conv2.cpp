#include "layer_dims.h"

//16 -> 32, k=3, pad=1
void conv2_1d(const data_t inputs[C2_IC][C2_IN_LEN],
              data_t outputs[C2_OC][C2_OUT_LEN],
              const data_t weights[C2_OC][C2_IC][C2_K],
              const data_t bias[C2_OC]) {

    #pragma HLS ARRAY_PARTITION variable=weights dim=3 type=complete
    #pragma HLS ARRAY_PARTITION variable=inputs dim=2 type=cyclic factor=3

    for (int oc = 0; oc < C2_OC; ++oc) {
        #pragma HLS PIPELINE off
        for (int o = 0; o < C2_OUT_LEN; ++o) {
            acc_t acc = bias[oc];

            for (int ic = 0; ic < C2_IC; ++ic) {
                for (int k = 0; k < C2_K; ++k) {

                    // #pragma HLS UNROLL
                    #pragma HLS PIPELINE II=1

                    int idx = o - C2_PAD + k;
                    data_t v = (idx >= 0 && idx < C2_IN_LEN) ? inputs[ic][idx] : (data_t)0;
                    acc += v * weights[oc][ic][k];
                }
            }
            outputs[oc][o] = (data_t)acc;
        }
    }
}