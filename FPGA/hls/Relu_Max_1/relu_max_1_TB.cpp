#include <algorithm>
#include <cstdio>
#include <cmath>
#include "layer_dims.h"

//decare relu_max_1 function
void relu_max_1(const data_t input[P1_C][P1_IN_LEN],
                data_t output[P1_C][P1_OUT_LEN]);

//load flat file
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

//storage for flat files
static float in_flat[P1_C * P1_IN_LEN];
static float golden_flat[P1_C * P1_OUT_LEN];

//shaped storage
static data_t input[P1_C][P1_IN_LEN];
static data_t output[P1_C][P1_OUT_LEN];

//tolerance stays tight
const float REL_TOL = 1e-6f;
const float ABS_TOL = 1e-7f;

//dump the chain input for conv2
static const char *CONV2_DIR = "C:/Users/cocol/Ruby_Proj/workspace/HLS/conv2/conv2_goldens/";
static void dump_chain(const char *name) {
    char path[512];
    snprintf(path, sizeof(path), "%sconv2_%s_chain_input.dat", CONV2_DIR, name);
    FILE *f = fopen(path, "w");
    if (!f) { printf("  WARN: cannot dump chain input %s\n", path); return; }
    for (int c = 0; c < P1_C; c++)                  // (16,16) channel-major, conv2's layout
        for (int o = 0; o < P1_OUT_LEN; o++)
            fprintf(f, "%.9g\n", (double)(float)output[c][o]);
    fclose(f);
}

//self-check the output against the input
static int selfcheck(void) {
    int fail = 0;
    for (int r = 0; r < P1_C; r++) {
        for (int o = 0; o < P1_OUT_LEN; o++) {
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
    snprintf(path_in, sizeof(path_in), "%spool1_%s_chain_input.dat", dir, name);
    golden_fatal = 0;             // golden was pooled from FLOAT conv1 out; input is now quantized
#else
    snprintf(path_in, sizeof(path_in), "%s%s", dir, input_file);
#endif
#ifdef USE_FIXED
    golden_fatal = 0;
#endif
    snprintf(path_out, sizeof(path_out), "%s%s", dir, golden_file);

    if (load_flat(path_in,  in_flat,     P1_C * P1_IN_LEN))  return 1;
    if (load_flat(path_out, golden_flat, P1_C * P1_OUT_LEN)) return 1;

    for (int i = 0; i < P1_C * P1_IN_LEN; ++i) {
        int c   = i / P1_IN_LEN;
        int in_ = i - P1_IN_LEN * c;
        input[c][in_] = in_flat[i];
    }

    relu_max_1(input, output);

    dump_chain(name);   // pooled output -> conv2's input file

    int   fail = 0;
    float worst = 0.0f;
    int   worst_c = -1, worst_o = -1;

    for (int c = 0; c < P1_C; c++) {
        for (int o = 0; o < P1_OUT_LEN; o++) {
            float got = output[c][o];
            float exp = golden_flat[c * P1_OUT_LEN + o];
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

    int sc = selfcheck();

    printf("[%s] checked %d values, %d failures, worst severity %.3f at [%d][%d], selfcheck %d%s\n",
           name, P1_C * P1_OUT_LEN, fail, worst, worst_c, worst_o, sc,
           golden_fatal ? "" : " (chain mode: golden non-fatal, selfcheck still enforced)");
    int failed = (golden_fatal && fail) || sc;
    printf("[%s] %s\n\n", name, failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}

int main() {
    const char *dir = "C:/Users/cocol/Ruby_Proj/workspace/HLS/Relu_Max_1/pool1_goldens/";

    int overall_fail = 0;
    overall_fail |= run_case("synth",        dir, "pool1_synth_input.dat",        "pool1_synth_golden_output.dat");
    overall_fail |= run_case("idx0",         dir, "pool1_idx0_input.dat",         "pool1_idx0_golden_output.dat");
    // overall_fail |= run_case("max_range",    dir, "pool1_max_range_input.dat",    "pool1_max_range_golden_output.dat");
    overall_fail |= run_case("first_intent", dir, "pool1_first_intent_input.dat", "pool1_first_intent_golden_output.dat");
    overall_fail |= run_case("ab156_intent", dir, "pool1_ab156_intent_input.dat", "pool1_ab156_intent_golden_output.dat");

    printf("=================================================\n");
    printf("OVERALL: %s\n", overall_fail ? "FAILED" : "PASSED");
    return overall_fail ? 1 : 0;
}