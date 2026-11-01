#include <hls_math.h>
#include "layer_dims.h"

//compute hyperbolic tan functions
static inline data_t tanh_act(data_t x) {
    return (data_t)hls::tanh(x.to_half());
}

//sigmoid(x) = 0.5 + 0.5*tanh(0.5*x)
static inline data_t sigmoid(data_t x) {
    return (data_t)0.5 + (data_t)0.5 * tanh_act((data_t)0.5 * x);
}

//one timestep of lstm
void LSTM_step(
    const data_t W_f[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_f[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_f[LSTM_HIDDEN],
    const data_t W_i[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_i[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_i[LSTM_HIDDEN],
    const data_t W_g[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_g[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_g[LSTM_HIDDEN],
    const data_t W_o[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_o[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_o[LSTM_HIDDEN],
    const data_t x[LSTM_IN_DIM],
    data_t h[LSTM_HIDDEN],     // in/out: short-term memory
    data_t c[LSTM_HIDDEN]      // in/out: long-term memory (cell state)
) {

    data_t f[LSTM_HIDDEN], i[LSTM_HIDDEN], g[LSTM_HIDDEN], o[LSTM_HIDDEN]; //vectors for f g i o gates

    GATES: for (int j = 0; j < LSTM_HIDDEN; j++) {
        // compute the four gates for the current timestep
        #pragma HLS PIPELINE off

        //forget gate: f[j] = sigmoid( W_f[j]·x + U_f[j]·h + b_f[j] )
        acc_t acc_f = b_f[j];
        for (int k = 0; k < LSTM_IN_DIM; k++) {
            #pragma HLS PIPELINE II=1
            acc_f += W_f[j][k] * x[k];
        }
        for (int m = 0; m < LSTM_HIDDEN; m++) {
            #pragma HLS PIPELINE II=1
            acc_f += U_f[j][m] * h[m];
        }
        f[j] = sigmoid((data_t)acc_f);

        //input gate: i[j] = sigmoid( W_i[j]·x + U_i[j]·h + b_i[j] )
        acc_t acc_i = b_i[j];
        for (int k = 0; k < LSTM_IN_DIM; k++) {
            #pragma HLS PIPELINE II=1
            acc_i += W_i[j][k] * x[k];
        }
        for (int m = 0; m < LSTM_HIDDEN; m++) {
            #pragma HLS PIPELINE II=1
            acc_i += U_i[j][m] * h[m];
        }
        i[j] = sigmoid((data_t)acc_i);

        //candidate: g[j] = tanh( W_g[j]·x + U_g[j]·h + b_g[j] )
        acc_t acc_g = b_g[j];
        for (int k = 0; k < LSTM_IN_DIM; k++) {
            #pragma HLS PIPELINE II=1
            acc_g += W_g[j][k] * x[k];
        }
        for (int m = 0; m < LSTM_HIDDEN; m++) {
            #pragma HLS PIPELINE II=1
            acc_g += U_g[j][m] * h[m];
        }
        g[j] = tanh_act((data_t)acc_g);

        //output gate: o[j] = sigmoid( W_o[j]·x + U_o[j]·h + b_o[j] )
        acc_t acc_o = b_o[j];
        for (int k = 0; k < LSTM_IN_DIM; k++) {
            #pragma HLS PIPELINE II=1
            acc_o += W_o[j][k] * x[k];
        }
        for (int m = 0; m < LSTM_HIDDEN; m++) {
            #pragma HLS PIPELINE II=1
            acc_o += U_o[j][m] * h[m];
        }
        o[j] = sigmoid((data_t)acc_o);
    }

    //update memory
    UPDATE: for (int j = 0; j < LSTM_HIDDEN; j++) {
        #pragma HLS PIPELINE off
        c[j] = f[j] * c[j] + i[j] * g[j]; //combine old cell state with new candidate
        h[j] = o[j] * tanh_act(c[j]);//updated memory to the hidden state to be passed forward
    }
}

//all 8 timesteps
void LSTM(
    const data_t W_f[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_f[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_f[LSTM_HIDDEN],
    const data_t W_i[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_i[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_i[LSTM_HIDDEN],
    const data_t W_g[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_g[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_g[LSTM_HIDDEN],
    const data_t W_o[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_o[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_o[LSTM_HIDDEN],
    const data_t x[LSTM_T][LSTM_IN_DIM],
    data_t h[LSTM_HIDDEN],     // out: final hidden state (h_n)
    data_t c[LSTM_HIDDEN]      // out: final cell state
) {

    INIT: for (int j = 0; j < LSTM_HIDDEN; j++) {
        h[j] = 0;
        c[j] = 0;
    }

    TIME: for (int t = 0; t < LSTM_T; t++) {
        #pragma HLS PIPELINE off
        LSTM_step(W_f, U_f, b_f,
                  W_i, U_i, b_i,
                  W_g, U_g, b_g,
                  W_o, U_o, b_o,
                  x[t], h, c);
    }

    // final output vector is in h
}