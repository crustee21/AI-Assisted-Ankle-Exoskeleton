from pathlib import Path
import hashlib
import shutil
import numpy as np
import torch
from model import FootDropCNN

REPO_ROOT  = Path(__file__).resolve().parent
DATA_DIR   = REPO_ROOT / "enabl3s_dataset_sliding"
WEIGHTS    = REPO_ROOT / "best_model.pt"
OUT_DIR    = REPO_ROOT / "golden_vectors"
FMT        = "%.8e"

HLS_ROOT   = REPO_ROOT.parent / "HLS"
HDR_PATH   = HLS_ROOT / "header" / "footdrop_weights.h"
HDR_FMT    = "%.9g"


def dump(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(arr).ravel(order="C").astype(np.float64), fmt=FMT)


def pick_cases(inputs, labels, subj):
    amax = np.abs(inputs).reshape(len(inputs), -1).max(axis=1)
    idxs = [("idx0", 0, "first window"),
            ("first_intent", int(np.where(labels == 1)[0][0]), "first intent (label 1)")]
    ab156 = np.where((labels == 1) & (subj == "AB156"))[0]
    if len(ab156):
        idxs.append(("ab156_intent", int(ab156[0]), "AB156 intent (different subject, small range)"))
    idxs.append(("max_range", int(amax.argmax()), "largest |x| (accumulator stress)"))
    out = [(n, inputs[i].astype(np.float32),
            f"{why}: idx {i}, subj {subj[i]}, label {int(labels[i])}") for n, i, why in idxs]
    rng = np.random.default_rng(42)
    out.append(("synth", rng.standard_normal((2, 32)).astype(np.float32),
                "synthetic seeded-random (rng=42)"))
    return out


def forward_stages(model, x_np):
    x = torch.from_numpy(x_np).unsqueeze(0)
    with torch.no_grad():
        c1 = model.conv1(x)
        p1 = model.pool(model.relu(c1))
        c2 = model.conv2(p1)
        p2 = model.pool(model.relu(c2))
        seq = p2.permute(0, 2, 1)
        _, (h_n, _) = model.lstm(seq)
        h = h_n.squeeze(0)
        logit = model.fc(h)
        assert torch.allclose(logit, model(x), atol=1e-6), "manual forward != model.forward"
    sq = lambda t: t.squeeze(0).numpy()
    return {"input": x_np, "conv1": sq(c1), "pool1": sq(p1), "conv2": sq(c2),
            "pool2": sq(p2), "lstm_in": sq(seq), "lstm_h": sq(h), "logit": sq(logit)}


def build_layers(sd):
    H = sd["lstm.bias_ih_l0"].shape[0] // 4
    lstm_params = {}
    for gi, g in enumerate(("i", "f", "g", "o")):
        s = slice(gi * H, (gi + 1) * H)
        lstm_params[f"w_ih_{g}"] = sd["lstm.weight_ih_l0"][s]
        lstm_params[f"w_hh_{g}"] = sd["lstm.weight_hh_l0"][s]
        lstm_params[f"b_{g}"]    = sd["lstm.bias_ih_l0"][s] + sd["lstm.bias_hh_l0"][s]

    return [
        dict(name="conv1", in_key="input",   out_key="conv1",
             params={"weights": sd["conv1.weight"], "bias": sd["conv1.bias"]}),
        dict(name="pool1", in_key="conv1",   out_key="pool1", params={}),
        dict(name="conv2", in_key="pool1",   out_key="conv2",
             params={"weights": sd["conv2.weight"], "bias": sd["conv2.bias"]}),
        dict(name="pool2", in_key="conv2",   out_key="pool2", params={}),
        dict(name="lstm",  in_key="lstm_in", out_key="lstm_h", params=lstm_params,
             dir="lstm_"),
        dict(name="fc",    in_key="lstm_h",  out_key="logit",
             params={"weights": sd["fc.weight"], "bias": sd["fc.bias"]}),
        dict(name="inference", in_key="input", out_key="logit", params={},
             dir="inference_goldens"),
    ]


LAYER_SPEC = {
"conv1": """conv1  (HLS: Conv1d only, pre-activation)
  input  (2, 32)     conv1_<case>_input.dat           idx = ch*32 + t
  output (16, 32)    conv1_<case>_golden_output.dat   idx = oc*32 + t
  weights(16, 2, 5)  conv1_weights.dat                idx = oc*(2*5)+ic*5+k
  bias   (16,)       conv1_bias.dat
  out[oc,t] = bias[oc] + sum_ic sum_k in[ic, t-2+k]*w[oc,ic,k]   (zero-pad, stride 1)""",
"pool1": """pool1  (HLS: ReLU then MaxPool1d/2, no params)
  input  (16, 32)    pool1_<case>_input.dat  (= conv1 output)
  output (16, 16)    pool1_<case>_golden_output.dat   idx = oc*16 + t
  out[oc,t] = max( max(in[oc,2t],0), max(in[oc,2t+1],0) )""",
"conv2": """conv2  (HLS: Conv1d only, pre-activation)
  input  (16, 16)    conv2_<case>_input.dat  (= pool1 output)   idx = ic*16 + t
  output (32, 16)    conv2_<case>_golden_output.dat             idx = oc*16 + t
  weights(32,16, 3)  conv2_weights.dat                idx = oc*(16*3)+ic*3+k
  bias   (32,)       conv2_bias.dat
  out[oc,t] = bias[oc] + sum_ic sum_k in[ic, t-1+k]*w[oc,ic,k]   (zero-pad, stride 1)""",
"pool2": """pool2  (HLS: ReLU then MaxPool1d/2, no params)
  input  (32, 16)    pool2_<case>_input.dat  (= conv2 output)
  output (32, 8)     pool2_<case>_golden_output.dat   idx = oc*8 + t
  out[oc,t] = max( max(in[oc,2t],0), max(in[oc,2t+1],0) )""",
"lstm": """lstm  (HLS: single-layer LSTM, 8 timesteps, h_0=c_0=0)
  input  (8, 32)     lstm_<case>_input.dat   idx = t*32 + f   (timestep-major, post-permute)
  output (32,)       lstm_<case>_golden_output.dat  (final hidden h_7; cell state c NOT dumped)

  Weights are split PER GATE (gate in i,f,g,o); each matrix stored row-major,
  row = output unit, col = input feature (idx = unit*width + col):
    w_ih_<gate> (32,32)  lstm_w_ih_<gate>.dat     (input->gate)
    w_hh_<gate> (32,32)  lstm_w_hh_<gate>.dat     (recurrent h->gate)
    b_<gate>    (32,)    lstm_b_<gate>.dat        = bias_ih + bias_hh  (the two PyTorch
                                                    biases are folded into one)
  for t in 0..7:  x = input[t]   (32-vec)
    i = sigmoid(w_ih_i @ x + w_hh_i @ h + b_i)
    f = sigmoid(w_ih_f @ x + w_hh_f @ h + b_f)
    g = tanh   (w_ih_g @ x + w_hh_g @ h + b_g)
    o = sigmoid(w_ih_o @ x + w_hh_o @ h + b_o)
    c = f*c + i*g;   h = o*tanh(c)
  output = h  (after t=7).   PyTorch gate order is i,f,g,o (input,forget,cell,output).""",
"fc": """fc  (HLS: Linear, produces the logit)
  input  (32,)       fc_<case>_input.dat  (= lstm output)
  output (1,)        fc_<case>_golden_output.dat
  weights(1,32)      fc_weights.dat       bias (1,) fc_bias.dat
  logit = weights @ input + bias;   probability = 1/(1+exp(-logit))""",
"inference": """inference  (HLS: the full chain, conv1 -> pool1 -> conv2 -> pool2 -> lstm -> fc)
  input  (2, 32)     inference_<case>_input.dat          idx = ch*32 + t
                                                         (identical to conv1_<case>_input.dat)
  output (1,)        inference_<case>_golden_output.dat  the logit
                                                         (identical to fc_<case>_golden_output.dat)

  No weight files here: the integrated top level reads its weights from the
  generated ROM header HLS/header/footdrop_weights.h, not from .dat files.

  This is the END-TO-END reference. The intermediate tensors never leave the DUT,
  so the only observable is the logit and the only invariant that must hold is
  sign(logit) == sign(golden). Magnitude drifts ~5e-3 through the quantized chain
  at <16,6>; that is expected and non-fatal. A sign disagreement is a decision
  flip and is always fatal.

  max_range is exported but is NOT part of the sign-off set: its input reaches
  |142| and intentionally overflows <16,6>. It is the range canary.""",
}


HLS_MIRROR = {
    "conv1":            "conv1/conv1_goldens",
    "pool1":            "Relu_Max_1/pool1_goldens",
    "conv2":            "conv2/conv2_goldens",
    "pool2":            "Relu_Max_2/pool2_goldens",
    "lstm_":            "LSTM/lstm_",
    "fc":               "linear/fc_goldens",
    "inference_goldens": "inference/inference_goldens",
}


def _c_init(arr, indent=2):
    arr = np.asarray(arr, dtype=np.float64)
    pad = " " * indent
    if arr.ndim == 1:
        return "{ " + ", ".join(HDR_FMT % v for v in arr) + " }"
    parts = [_c_init(sub, indent + 2) for sub in arr]
    return "{\n" + pad + (",\n" + pad).join(parts) + "\n" + " " * (indent - 2) + "}"


def emit_weight_header(sd, lstm_params, logits, path=HDR_PATH):
    digest = hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()[:16]

    decls = [
        ("conv1_w", "[C1_OC][C1_IC][C1_K]", sd["conv1.weight"]),
        ("conv1_b", "[C1_OC]",              sd["conv1.bias"]),
        ("conv2_w", "[C2_OC][C2_IC][C2_K]", sd["conv2.weight"]),
        ("conv2_b", "[C2_OC]",              sd["conv2.bias"]),
    ]
    for g in ("i", "f", "g", "o"):
        decls += [
            (f"lstm_w_ih_{g}", "[LSTM_HIDDEN][LSTM_IN_DIM]", lstm_params[f"w_ih_{g}"]),
            (f"lstm_w_hh_{g}", "[LSTM_HIDDEN][LSTM_HIDDEN]", lstm_params[f"w_hh_{g}"]),
            (f"lstm_b_{g}",    "[LSTM_HIDDEN]",              lstm_params[f"b_{g}"]),
        ]
    decls.append(("fc_w", "[FC_IN]", np.asarray(sd["fc.weight"]).reshape(-1)))

    n_params = sum(int(np.asarray(a).size) for _, _, a in decls) + 1
    body = "\n\n".join(
        f"static const data_t {name}{dims} = {_c_init(arr)};" for name, dims, arr in decls
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""\
#ifndef FOOTDROP_WEIGHTS_H
#define FOOTDROP_WEIGHTS_H

#include "layer_dims.h"

{body}

static const data_t fc_b = {HDR_FMT % float(np.asarray(sd["fc.bias"]).reshape(-1)[0])};

#endif
""")
    return path, n_params, digest


def selfcheck_conv1(x, w, b, out):
    K, P, L = w.shape[2], w.shape[2] // 2, x.shape[1]
    ref = np.zeros_like(out)
    for oc in range(w.shape[0]):
        for t in range(L):
            acc = b[oc]
            for ic in range(x.shape[0]):
                for k in range(K):
                    if 0 <= t - P + k < L:
                        acc += x[ic, t - P + k] * w[oc, ic, k]
            ref[oc, t] = acc
    return np.abs(ref - out).max()


def main():
    if not WEIGHTS.exists():
        raise SystemExit(f"{WEIGHTS.name} not found — train the model first (python train.py).")

    model = FootDropCNN(c1=16, c2=32, lstm_hidden=32)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.eval()

    inputs = np.load(DATA_DIR / "inputs.npy")
    labels = np.load(DATA_DIR / "labels.npy")
    subj   = np.load(DATA_DIR / "subject_ids.npy", allow_pickle=True)
    assert inputs.shape[1:] == (2, 32), f"expected (N,2,32), got {inputs.shape}"

    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    layers = build_layers(sd)

    for L in layers:
        d = OUT_DIR / L.get("dir", L["name"])
        d.mkdir(parents=True, exist_ok=True)
        for pname, arr in L["params"].items():
            dump(d / f"{L['name']}_{pname}.dat", arr)
        (d / f"{L['name']}_spec.txt").write_text(LAYER_SPEC[L["name"]] + "\n")

    st0 = forward_stages(model, inputs[0].astype(np.float32))
    err = selfcheck_conv1(st0["input"], sd["conv1.weight"], sd["conv1.bias"], st0["conv1"])
    print(f"conv1 layout self-check: max|naive - torch| = {err:.2e}  ({'OK' if err < 1e-3 else 'FAIL'})")

    manifest = []
    logits = []
    for cname, x_np, src in pick_cases(inputs, labels, subj):
        st = forward_stages(model, x_np)
        for L in layers:
            d = OUT_DIR / L.get("dir", L["name"])
            dump(d / f"{L['name']}_{cname}_input.dat",         st[L["in_key"]])
            dump(d / f"{L['name']}_{cname}_golden_output.dat", st[L["out_key"]])
        logit = float(st["logit"].item())
        logits.append((cname, logit))
        manifest.append(f"{cname:14s}  {src}  | logit={logit:.5f}  p={1/(1+np.exp(-logit)):.4f}")
        print(f"  {cname:14s} logit={logit:+.4f}")

    (OUT_DIR / "manifest.txt").write_text("\n".join(manifest) + "\n")
    print(f"\nDone -> {OUT_DIR}   layers: {', '.join(L['name'] for L in layers)}")

    lstm_params = next(L for L in layers if L["name"] == "lstm")["params"]
    hdr, n_params, digest = emit_weight_header(sd, lstm_params, logits)
    print(f"Weight ROM header -> {hdr}   ({n_params} params, ckpt {digest})")

    n_copied = 0
    for src_dir, dst_rel in HLS_MIRROR.items():
        src = OUT_DIR / src_dir
        dst = HLS_ROOT / dst_rel
        if not src.is_dir():
            print(f"  WARNING: {src} missing, not mirrored")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.iterdir()):
            if f.is_file():
                shutil.copy2(f, dst / f.name)
                n_copied += 1
    print(f"Mirrored {n_copied} golden files into {HLS_ROOT}")


if __name__ == "__main__":
    main()
