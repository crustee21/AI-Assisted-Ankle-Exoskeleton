
#include "layer_dims.h"

//dot product of the input vector and the weight vector plus bias
void fc(const data_t inputs[FC_IN],
        const data_t weights[FC_IN],
        data_t bias,
        data_t outputs[FC_OUT]) {

    acc_t acc = bias;
    FC_MAC: for (int j = 0; j < FC_IN; j++) {
        #pragma HLS PIPELINE off
        acc += inputs[j] * weights[j];
    }

    outputs[0] = (data_t)acc;
}
