#include <algorithm>
#include <cstdio>
#include <cmath>
#include "layer_dims.h"

//declaration of relu_max_2 function for the testbench
void relu_max_2(const data_t input[P2_C][P2_IN_LEN],
                data_t output[P2_C][P2_OUT_LEN]);

//flate file loader
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

//flat storage for input and golden output
static float in_flat[P2_C * P2_IN_LEN];
static float golden_flat[P2_C * P2_OUT_LEN];

//reshaped storage
static data_t input[P2_C][P2_IN_LEN];
static data_t output[P2_C][P2_OUT_LEN];

//tight tolerance for comparison
const float ABS_TOL = 1e-7f;

//dump the chain input for the LSTM testbench
static const char *LSTM_DIR = "C:/Users/cocol/Ruby_Proj/workspace/HLS/LSTM/lstm_/";
static void dump_chain(const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%slstm_%s_chain_input.dat", LSTM_DIR, name);
    FILE *f = fopen(path, "w");
    if (!f) { printf("  WARN: cannot dump chain input %s\n", path); return; }
    for (int t = 0; t < P2_OUT_LEN; t++)            // T = 8 timesteps
        for (int c = 0; c < P2_C; c++)              // 32 features per timestep
            fprintf(f, "%.9g\n", (double)(float)output[c][t]);
    fclose(f);
}

//self-check the output against the input to ensure correctness
static int selfcheck(void) {
    int fail = 0;
    for (int r = 0; r < P2_C; r++) {
        for (int o = 0; o < P2_OUT_LEN; o++) {
            float a = input[r][2*o], b = input[r][2*o + 1];
            float m = a > b ? a : b;
            float exp = m > 0.0f ? m : 0.0f;
            if ((float)output[r][o] != exp) {          // cast: ap_fixed-vs-float compare is ambiguous under Vitis clang
                if (fail < 10)
                    printf("  SELFCHECK r=%d o=%d got=%.6f exp=%.6f (a=%.6f b=%.6f)\n",
                           r, o, (double)output[r][o], exp, a, b);
                ++fail;
            }
        }
    }
    return fail;
}

static int run_case(const char *name, const char *dir,
                    const char *input_file, const char *golden_file) {
    char path_in[512], path_out[512];
    int golden_fatal = 1;
#ifdef CHAIN_INPUT
    snprintf(path_in, sizeof(path_in), "%spool2_%s_chain_input.dat", dir, name);
    golden_fatal = 0;
#else
    snprintf(path_in, sizeof(path_in), "%s%s", dir, input_file);
#endif
#ifdef USE_FIXED
    golden_fatal = 0;
#endif
    snprintf(path_out, sizeof(path_out), "%s%s", dir, golden_file);

    if (load_flat(path_in,  in_flat,     P2_C * P2_IN_LEN))  return 1;
    if (load_flat(path_out, golden_flat, P2_C * P2_OUT_LEN)) return 1;

    for (int i = 0; i < P2_C * P2_IN_LEN; ++i) {
        int c   = i / P2_IN_LEN;
        int in_ = i - P2_IN_LEN * c;
        input[c][in_] = in_flat[i];
    }

    relu_max_2(input, output);

    dump_chain(name);   // pooled + permuted output -> LSTM's input file

    int   fail = 0;
    float worst = 0.0f;
    int   worst_c = -1, worst_o = -1;

    for (int c = 0; c < P2_C; c++) {
        for (int o = 0; o < P2_OUT_LEN; o++) {
            float got = output[c][o];
            float exp = golden_flat[c * P2_OUT_LEN + o];
            float diff = std::abs(got - exp);
            float tol  = ABS_TOL + REL_TOL * std::abs(exp);
            float severity = diff / tol;
            if (severity > worst) { worst = severity; worst_c = c; worst_o = o; }
            if (diff > tol) {
                if (fail < 10)
                    printf("  MISMATCH c=%d o=%d got=%.6f exp=%.6f diff=%.3e tol=%.3e\n",
                           c, o, got, exp, diff, tol);
                ++fail;
            }
        }
    }

    //make sure no negatives
    int neg = 0;
    for (int c = 0; c < P2_C; c++)
        for (int o = 0; o < P2_OUT_LEN; o++)
            if ((float)output[c][o] < 0.0f) ++neg;

    int sc = selfcheck();

    printf("[%s] checked %d values, %d failures, worst severity %.3f at [%d][%d], selfcheck %d, negatives %d%s\n",
           name, P2_C * P2_OUT_LEN, fail, worst, worst_c, worst_o, sc, neg,
           golden_fatal ? "" : " (chain mode: golden non-fatal, selfcheck+neg still enforced)");
    int failed = (golden_fatal && fail) || sc || neg;
    printf("[%s] %s\n\n", name, failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}

int main() {
    const char *dir = "C:/Users/cocol/Ruby_Proj/workspace/HLS/Relu_Max_2/pool2_goldens/";

    int overall_fail = 0;
    overall_fail |= run_case("synth",        dir, "pool2_synth_input.dat",        "pool2_synth_golden_output.dat");
    overall_fail |= run_case("idx0",         dir, "pool2_idx0_input.dat",         "pool2_idx0_golden_output.dat");
    // overall_fail |= run_case("max_range",    dir, "pool2_max_range_input.dat",    "pool2_max_range_golden_output.dat");
    overall_fail |= run_case("first_intent", dir, "pool2_first_intent_input.dat", "pool2_first_intent_golden_output.dat");
    overall_fail |= run_case("ab156_intent", dir, "pool2_ab156_intent_input.dat", "pool2_ab156_intent_golden_output.dat");

    printf("=================================================\n");
    printf("OVERALL: %s\n", overall_fail ? "FAILED" : "PASSED");
    return overall_fail ? 1 : 0;
}