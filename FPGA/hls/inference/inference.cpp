#include "layer_dims.h"
#include "footdrop_weights.h"   // generated

//declarations of the layer functions
void conv1_1d(const data_t inputs[C1_IC][C1_IN_LEN],
              data_t outputs[C1_OC][C1_OUT_LEN],
              const data_t weights[C1_OC][C1_IC][C1_K],
              const data_t bias[C1_OC]);

void relu_max_1(const data_t input[P1_C][P1_IN_LEN],
                data_t output[P1_C][P1_OUT_LEN]);

void conv2_1d(const data_t inputs[C2_IC][C2_IN_LEN],
              data_t outputs[C2_OC][C2_OUT_LEN],
              const data_t weights[C2_OC][C2_IC][C2_K],
              const data_t bias[C2_OC]);

void relu_max_2(const data_t input[P2_C][P2_IN_LEN],
                data_t output[P2_C][P2_OUT_LEN]);

void LSTM(const data_t W_f[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_f[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_f[LSTM_HIDDEN],
          const data_t W_i[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_i[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_i[LSTM_HIDDEN],
          const data_t W_g[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_g[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_g[LSTM_HIDDEN],
          const data_t W_o[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_o[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_o[LSTM_HIDDEN],
          const data_t x[LSTM_T][LSTM_IN_DIM],
          data_t h[LSTM_HIDDEN],
          data_t c[LSTM_HIDDEN]);

void fc(const data_t inputs[FC_IN],
        const data_t weights[FC_IN],
        data_t bias,
        data_t outputs[FC_OUT]);


//run cnn on 320 ms window
void inference(const data_t inputs[C1_IC][C1_IN_LEN],
               data_t outputs[FC_OUT]) {

    //CNN layer buffers
    data_t c1_out[C1_OC][C1_OUT_LEN];
    data_t p1_out[P1_C][P1_OUT_LEN];
    data_t c2_out[C2_OC][C2_OUT_LEN];
    data_t p2_out[P2_C][P2_OUT_LEN];
    data_t seq[LSTM_T][LSTM_IN_DIM];
    data_t h[LSTM_HIDDEN];
    data_t c[LSTM_HIDDEN];      // final cell state

    conv1_1d(inputs, c1_out, conv1_w, conv1_b);
    relu_max_1(c1_out, p1_out);
    conv2_1d(p1_out, c2_out, conv2_w, conv2_b);
    relu_max_2(c2_out, p2_out);

    //turn 32x8 to 8x32
    PERMUTE_T: for (int t = 0; t < LSTM_T; t++) {
        PERMUTE_CH: for (int ch = 0; ch < LSTM_IN_DIM; ch++) {
            seq[t][ch] = p2_out[ch][t];
        }
    }

    //process 8 timestamps and output h
    LSTM(lstm_w_ih_f, lstm_w_hh_f, lstm_b_f,
         lstm_w_ih_i, lstm_w_hh_i, lstm_b_i,
         lstm_w_ih_g, lstm_w_hh_g, lstm_b_g,
         lstm_w_ih_o, lstm_w_hh_o, lstm_b_o,
         seq, h, c);

    fc(h, fc_w, fc_b, outputs); //compute logit
}


//maintain sliding window
void inference_stream(data_t emg_sample, data_t gyro_sample, data_t *logit) {

    #pragma HLS INTERFACE mode=ap_none port=emg_sample
    #pragma HLS INTERFACE mode=ap_none port=gyro_sample
    #pragma HLS INTERFACE mode=s_axilite port=logit bundle=ctrl
    #pragma HLS INTERFACE mode=ap_ctrl_hs port=return

    //sliding window buffer
    static data_t window[C1_IC][C1_IN_LEN] = {};

    //shift the window to make room for the new sample
    SHIFT: for (int t = 0; t < C1_IN_LEN - 1; t++) {
        #pragma HLS PIPELINE off
        window[0][t] = window[0][t + 1];
        window[1][t] = window[1][t + 1];
    }
    window[0][C1_IN_LEN - 1] = emg_sample;
    window[1][C1_IN_LEN - 1] = gyro_sample;

    data_t out[FC_OUT];
    inference(window, out); //process newest data
    *logit = out[0]; //store prediction
}
