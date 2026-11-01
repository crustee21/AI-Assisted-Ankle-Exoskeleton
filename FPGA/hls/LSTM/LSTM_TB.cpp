#include <algorithm>
#include <cstdio>
#include <cmath>
#include "layer_dims.h"

//declare LSTM function
void LSTM(
    const data_t W_f[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_f[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_f[LSTM_HIDDEN],
    const data_t W_i[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_i[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_i[LSTM_HIDDEN],
    const data_t W_g[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_g[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_g[LSTM_HIDDEN],
    const data_t W_o[LSTM_HIDDEN][LSTM_IN_DIM], const data_t U_o[LSTM_HIDDEN][LSTM_HIDDEN], const data_t b_o[LSTM_HIDDEN],
    const data_t x[LSTM_T][LSTM_IN_DIM],
    data_t h[LSTM_HIDDEN],
    data_t c[LSTM_HIDDEN]);

//read and store data to be used
static int load_flat(const char *path, float *buf, int n) {
    FILE *f = fopen(path, "r");
    if (!f) { printf("ERROR opening %s\n", path); return 1; }
    for (int i = 0; i < n; i++) {
        if (fscanf(f, "%f", &buf[i]) != 1) {
            printf("ERROR: %s truncated at %d\n", path, i);
            fclose(f);
            return 1;
        }
    }
    fclose(f);
    return 0;
}

//store lstm learned parameters
static data_t W_f[LSTM_HIDDEN][LSTM_IN_DIM], U_f[LSTM_HIDDEN][LSTM_HIDDEN], b_f[LSTM_HIDDEN];
static data_t W_i[LSTM_HIDDEN][LSTM_IN_DIM], U_i[LSTM_HIDDEN][LSTM_HIDDEN], b_i[LSTM_HIDDEN];
static data_t W_g[LSTM_HIDDEN][LSTM_IN_DIM], U_g[LSTM_HIDDEN][LSTM_HIDDEN], b_g[LSTM_HIDDEN];
static data_t W_o[LSTM_HIDDEN][LSTM_IN_DIM], U_o[LSTM_HIDDEN][LSTM_HIDDEN], b_o[LSTM_HIDDEN];

//store input and output data for testing
static float  x_flat[LSTM_T * LSTM_IN_DIM];
static float  golden_flat[LSTM_HIDDEN];
static data_t x[LSTM_T][LSTM_IN_DIM];
static data_t h[LSTM_HIDDEN];
static data_t c[LSTM_HIDDEN];

#ifdef USE_FIXED
//tolerances. differences are due to precision and not mathematical errors
const float REL_TOL = 1e-2f;
const float ABS_TOL = 1.5e-2f;
#else
const float REL_TOL = 1e-3f;
const float ABS_TOL = 1e-5f;
#endif

//save lstm output to file
static const char *FC_DIR = "C:/Users/cocol/Ruby_Proj/workspace/HLS/linear/fc_goldens/";
static void dump_chain(const char *name, const data_t *h_out) {
    char path[512];
    snprintf(path, sizeof(path), "%sfc_%s_chain_input.dat", FC_DIR, name);
    FILE *f = fopen(path, "w");
    if (!f) { printf("  WARN: cannot dump chain input %s\n", path); return; }
    for (int j = 0; j < LSTM_HIDDEN; j++)
        fprintf(f, "%.9g\n", (double)(float)h_out[j]);
    fclose(f);
}

//reconstruct to 2d matrix
static int load_mat(const char *dir, const char *fname, data_t *dst, int rows, int cols) {
    char path[512];
    snprintf(path, sizeof(path), "%s%s", dir, fname);
    static float tmp[LSTM_HIDDEN * LSTM_IN_DIM];
    if (load_flat(path, tmp, rows * cols)) return 1;
    for (int r = 0; r < rows; r++)
        for (int cc = 0; cc < cols; cc++)
            dst[r * cols + cc] = tmp[r * cols + cc];
    return 0;
}

static int load_all_weights(const char *dir) {
    int e = 0;
    //load all LSTM weights and biases from files
    e |= load_mat(dir, "lstm_w_ih_f.dat", &W_f[0][0], LSTM_HIDDEN, LSTM_IN_DIM);
    e |= load_mat(dir, "lstm_w_hh_f.dat", &U_f[0][0], LSTM_HIDDEN, LSTM_HIDDEN);
    e |= load_mat(dir, "lstm_b_f.dat",    &b_f[0],    LSTM_HIDDEN, 1);
    e |= load_mat(dir, "lstm_w_ih_i.dat", &W_i[0][0], LSTM_HIDDEN, LSTM_IN_DIM);
    e |= load_mat(dir, "lstm_w_hh_i.dat", &U_i[0][0], LSTM_HIDDEN, LSTM_HIDDEN);
    e |= load_mat(dir, "lstm_b_i.dat",    &b_i[0],    LSTM_HIDDEN, 1);
    e |= load_mat(dir, "lstm_w_ih_g.dat", &W_g[0][0], LSTM_HIDDEN, LSTM_IN_DIM);
    e |= load_mat(dir, "lstm_w_hh_g.dat", &U_g[0][0], LSTM_HIDDEN, LSTM_HIDDEN);
    e |= load_mat(dir, "lstm_b_g.dat",    &b_g[0],    LSTM_HIDDEN, 1);
    e |= load_mat(dir, "lstm_w_ih_o.dat", &W_o[0][0], LSTM_HIDDEN, LSTM_IN_DIM);
    e |= load_mat(dir, "lstm_w_hh_o.dat", &U_o[0][0], LSTM_HIDDEN, LSTM_HIDDEN);
    e |= load_mat(dir, "lstm_b_o.dat",    &b_o[0],    LSTM_HIDDEN, 1);
    return e;
}

//run a single test case: load input, run LSTM, compare with golden output
static int run_case(const char *name, const char *dir,
                    const char *input_file, const char *golden_file) {
    char path_in[512], path_out[512];
#ifdef CHAIN_INPUT
    snprintf(path_in, sizeof(path_in), "%slstm_%s_chain_input.dat", dir, name);
    const int golden_fatal = 0;
#else
    snprintf(path_in, sizeof(path_in), "%s%s", dir, input_file);
    const int golden_fatal = 1;
#endif
    snprintf(path_out, sizeof(path_out), "%s%s", dir, golden_file);

    if (load_flat(path_in,  x_flat,      LSTM_T * LSTM_IN_DIM)) return 1;
    if (load_flat(path_out, golden_flat, LSTM_HIDDEN))          return 1;

    //reshape
    for (int t = 0; t < LSTM_T; t++)
        for (int d = 0; d < LSTM_IN_DIM; d++)
            x[t][d] = x_flat[t * LSTM_IN_DIM + d];

    LSTM(W_f, U_f, b_f, W_i, U_i, b_i, W_g, U_g, b_g, W_o, U_o, b_o, x, h, c); //execute

    dump_chain(name, h);   // final hidden state -> fc's input file

    int   fail = 0;
    float worst = 0.0f;
    int   worst_j = -1;

    for (int j = 0; j < LSTM_HIDDEN; j++) {
        float got = h[j];
        float exp = golden_flat[j];
        float diff = std::abs(got - exp);
        float tol  = ABS_TOL + REL_TOL * std::abs(exp);
        float severity = diff / tol;
        if (severity > worst) { worst = severity; worst_j = j; }
        if (diff > tol) {
            if (fail < 10)
                printf("  MISMATCH j=%d got=%.6f exp=%.6f diff=%.3e tol=%.3e\n",
                       j, got, exp, diff, tol);
            ++fail;
        }
    }

    int bounds = 0;
    for (int j = 0; j < LSTM_HIDDEN; j++) {
        if (!(std::abs((float)h[j]) <= 1.0f + 1e-4f)) {
            if (bounds < 5) printf("  BOUNDS h[%d]=%.6f outside [-1,1]\n", j, (double)h[j]);
            ++bounds;
        }
    }

    printf("[%s] checked %d values, %d failures, worst severity %.3f at h[%d], bounds %d%s\n",
           name, LSTM_HIDDEN, fail, worst, worst_j, bounds,
           golden_fatal ? "" : " (chain mode: golden non-fatal, bounds still enforced)");
    int failed = (golden_fatal && fail) || bounds;
    printf("[%s] %s\n\n", name, failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}

int main() {
    const char *dir = "C:/Users/cocol/Ruby_Proj/workspace/HLS/LSTM/lstm_/";

    if (load_all_weights(dir)) return 1;

    int overall_fail = 0;
    overall_fail |= run_case("synth",        dir, "lstm_synth_input.dat",        "lstm_synth_golden_output.dat");
    overall_fail |= run_case("idx0",         dir, "lstm_idx0_input.dat",         "lstm_idx0_golden_output.dat");
    // overall_fail |= run_case("max_range",    dir, "lstm_max_range_input.dat",    "lstm_max_range_golden_output.dat");
    overall_fail |= run_case("first_intent", dir, "lstm_first_intent_input.dat", "lstm_first_intent_golden_output.dat");
    overall_fail |= run_case("ab156_intent", dir, "lstm_ab156_intent_input.dat", "lstm_ab156_intent_golden_output.dat");

    printf("=================================================\n");
    printf("OVERALL: %s\n", overall_fail ? "FAILED" : "PASSED");
    return overall_fail ? 1 : 0;
}