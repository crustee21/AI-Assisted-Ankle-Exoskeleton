#ifndef LAYER_DIMS_H
#define LAYER_DIMS_H


//data_t = activations and weights
//acc_t = MAC accumulator only
#ifndef USE_FIXED
#define USE_FIXED
#endif
#include <ap_fixed.h>
//<16,6> = 1 sign + 5 int + 10 frac -> range [-32,32), resolution 2^-10 ~= 1e-3
typedef ap_fixed<16, 6, AP_RND, AP_SAT> data_t;
// Wider accumulator
typedef ap_fixed<32, 8> acc_t;

  // typedef float data_t; //floating point
  // typedef float acc_t;


//CNN layer dimensions
//pipeline: (2, 32) -> conv1 -> (16, 32) -> relu/pool -> (16, 16) -> conv2 -> (32, 16) -> relu/pool -> (32, 8) -> permute -> (8, 32) -> LSTM -> (32,) -> fc -> 1 logit

//conv1: conv1d(2 -> 16, k=5, pad=2), length is same (32)
constexpr int C1_IC      = 2;
constexpr int C1_IN_LEN  = 32;
constexpr int C1_OC      = 16;
constexpr int C1_OUT_LEN = 32;
constexpr int C1_K       = 5;
constexpr int C1_PAD     = C1_K / 2;

//pool1 for ReLU and MaxPool 1d, 32 -> 16
constexpr int P1_C       = C1_OC;
constexpr int P1_IN_LEN  = C1_OUT_LEN;
constexpr int P1_OUT_LEN = P1_IN_LEN / 2;

//conv2: conv1d(16 -> 32, k=3, pad=1), length is smae (16)
constexpr int C2_IC      = P1_C;
constexpr int C2_IN_LEN  = P1_OUT_LEN;
constexpr int C2_OC      = 32;
constexpr int C2_OUT_LEN = 16;
constexpr int C2_K       = 3;
constexpr int C2_PAD     = C2_K / 2;

//pool2 for ReLU and MaxPool 1d, 32 -> 8
constexpr int P2_C       = C2_OC;
constexpr int P2_IN_LEN  = C2_OUT_LEN;
constexpr int P2_OUT_LEN = P2_IN_LEN / 2;

//LSTM layer dimensions
constexpr int LSTM_T      = P2_OUT_LEN;   // 8 timesteps for loop bounds
constexpr int LSTM_IN_DIM = P2_C;         // 32 features per timestep
constexpr int LSTM_HIDDEN = 32;           // H

//Fully connected layer dimensions
constexpr int FC_IN  = LSTM_HIDDEN;       // 32 (= final h_n width)
constexpr int FC_OUT = 1;

#endif // LAYER_DIMS_H
