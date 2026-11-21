#!/usr/bin/env python3
"""Extract Gemmini-ready int8 parameters from a pt2e-quantized ExportedProgram (.pth) into a C header.

Gemmini has no activation zero points, so for every conv/linear the input and output zero
points (and float biases, as XNNPACKQuantizer leaves them) are folded into one int32 bias:
    y_q = clamp(round((sum(w_q * x_q) + bias_fold) * ACC_SCALE), -128, 127)
"""
import argparse
import functools

import numpy as np
import torch
import torchao.quantization.pt2e  # noqa: F401  registers quantized_decomposed ops

OPS = torch.ops.aten
QD = torch.ops.quantized_decomposed
Q, DQ = QD.quantize_per_tensor.default, QD.dequantize_per_tensor.default
CONV, LINEAR = OPS.conv2d.default, OPS.linear.default
RELU = (OPS.relu.default, OPS.relu_.default)
SHAPE_OPS = (OPS.flatten.using_ints, OPS.view.default, OPS.reshape.default, OPS.permute.default)

# Names used by the existing Gemmini C code (braggnn.h / braggnn_pt2e.h).
COMPAT_NAMES = {
    "cnn_layers.0": "conv1_weights_flat", "cnn_layers.2": "conv2_weights_flat",
    "cnn_layers.4": "conv3_weights_flat", "dense_layers.0": "fc1_weights",
    "dense_layers.2": "fc2_weights", "dense_layers.4": "fc3_weights",
    "dense_layers.6": "fc4_weights", "dense_layers.8": "output_weights",
}


def module_path(node):
    return list(node.meta["nn_module_stack"].values())[-1][0]


def only_user(node):
    users = list(node.users)
    assert len(users) == 1, f"{node.name}: expected 1 user, got {[u.name for u in users]}"
    return users[0]


def qparams(node):
    assert node.target in (Q, DQ), f"{node.name}: expected a (de)quantize node, got {node.target}"
    return float(node.args[1]), int(node.args[2])


def extract_compute(gm, node):
    const = lambda n: functools.reduce(getattr, n.target.split("."), gm).detach().numpy()

    x_s, x_zp = qparams(node.args[0])
    w_dq = node.args[1]
    w_s, w_zp = qparams(w_dq)
    assert w_zp == 0, f"{node.name}: weight zero point {w_zp} != 0"
    w_q = const(w_dq.args[0]).astype(np.int64)

    b = node.args[2] if len(node.args) > 2 else None
    if b is None:
        b_acc = np.zeros(w_q.shape[0])
    elif b.op == "get_attr":
        b_acc = const(b).astype(np.float64) / (x_s * w_s)
    else:
        b_acc = const(b.args[0]).astype(np.float64)

    user, fused_relu = only_user(node), False
    if user.target in RELU:
        user, fused_relu = only_user(user), True
    y_s, y_zp = qparams(user)

    acc_scale = x_s * w_s / y_s
    w_sum = w_q.reshape(w_q.shape[0], -1).sum(axis=1)
    bias = np.round(b_acc - x_zp * w_sum + y_zp / acc_scale).astype(np.int64)
    assert np.abs(bias).max() < 2**31, f"{node.name}: folded bias overflows int32"

    if not fused_relu:
        act = "NO_ACTIVATION"
    elif y_zp == 0:
        act = "RELU"
    elif y_zp == -128:
        act = "NO_ACTIVATION"  # real 0 maps to -128, so int8 saturation already is the ReLU
    else:
        raise ValueError(f"{node.name}: fused ReLU with output zero point {y_zp} can't run on Gemmini")

    if node.target == CONV:
        stride, padding = (list(node.args[3:5]) + [[1], [0]][len(node.args[3:5]):])
        assert all(s == 1 for s in stride) and all(p == 0 for p in padding), \
            f"{node.name}: only stride 1 / padding 0 is handled"
        oc, ic, kh, kw = w_q.shape
        w_out = w_q.transpose(2, 3, 1, 0).reshape(kh * kw * ic, oc)  # [KH*KW*IC][OC]
    else:
        w_out = w_q  # [OUT][IN], as in braggnn.h
    return dict(path=module_path(node), kind="conv" if node.target == CONV else "linear",
                w=w_out, bias=bias, acc_scale=acc_scale, act=act,
                x=(x_s, x_zp), w_s=w_s, y=(y_s, y_zp))


def extract_other(node):
    ins = [qparams(a) for a in node.args if hasattr(a, "target") and a.target == DQ]
    user = only_user(node)
    while user.target in SHAPE_OPS:
        user = only_user(user)
    extra = {"negative_slope": float(node.args[1]) if len(node.args) > 1 else 0.01} \
        if node.target == OPS.leaky_relu.default else {}
    return dict(path=module_path(node) or node.name, op=node.target.__name__.split(".")[0],
                ins=ins, out=qparams(user) if user.target == Q else None, **extra)


def c_name(path):
    return path.replace(".", "_").upper() or "MODEL"


def fmt_2d(a):
    return ",\n".join("  {" + ", ".join(f"{int(v):4d}" for v in row) + "}" for row in a)


def fmt_1d(a, per_line=8):
    return ",\n".join("  " + ", ".join(f"{int(v):10d}" for v in a[i:i + per_line])
                      for i in range(0, len(a), per_line))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pth", help="ExportedProgram saved by pt2e-quantize.py")
    ap.add_argument("-o", "--output", default="braggnn_params.h")
    ap.add_argument("--psz", type=int, default=11, help="patch size, for OUTPUT_DEQUANT_SCALE")
    args = ap.parse_args()

    gm = torch.export.load(args.pth).module()
    computes, others, input_q, output_q = [], [], None, None
    for node in gm.graph.nodes:
        if node.op == "output":
            output_q = node.args[0][0].args[0] if node.args[0][0].target == DQ else None
        if node.op != "call_function":
            continue
        if node.target == Q and node.args[0].op == "placeholder":
            input_q = node
        elif node.target in (CONV, LINEAR):
            computes.append(extract_compute(gm, node))
        elif node.target not in (Q, DQ, *RELU, *SHAPE_OPS):
            others.append(extract_other(node))
    assert input_q is not None and output_q is not None, "could not find input/output quantization"
    in_s, in_zp = qparams(input_q)
    out_s, out_zp = qparams(output_q)

    with open(args.output, "w") as f:
        f.write(f"// Generated by extract_quantized_params.py from {args.pth}\n")
        f.write("// conv/linear: y_q = clamp(round((sum(w_q * x_q) + bias) * ACC_SCALE), -128, 127)\n")
        f.write("// Zero points and biases are folded into bias; ACTIVATION is the Gemmini act to use.\n")
        f.write("#pragma once\n#include \"include/gemmini_testutils.h\"\n\n")

        f.write(f"#define INPUT_SCALE {in_s:.10e}f\n#define INPUT_ZERO_POINT {in_zp}\n")
        f.write(f"#define INPUT_DEQUANT_SCALE {1.0 / in_s:.10e}f  // x_q = round(x * this) + INPUT_ZERO_POINT\n")
        f.write(f"#define OUTPUT_SCALE {out_s:.10e}f\n#define OUTPUT_ZERO_POINT {out_zp}\n")
        f.write(f"#define OUTPUT_DEQUANT_SCALE {out_s * args.psz:.10e}f"
                "  // pixels = (y_q - OUTPUT_ZERO_POINT) * this\n\n")

        for L in computes:
            p = c_name(L["path"])
            suffix = "CONV_QUANT_ACC_SCALE" if L["kind"] == "conv" else "GEMM_ACC_SCALE"
            f.write(f"// /{L['path']} ({L['kind']}): x=(scale {L['x'][0]:.6e}, zp {L['x'][1]}), "
                    f"w_scale {L['w_s']:.6e}, y=(scale {L['y'][0]:.6e}, zp {L['y'][1]})\n")
            f.write(f"#define {p}_{suffix} {L['acc_scale']:.10e}f\n")
            f.write(f"#define {p}_ACTIVATION {L['act']}\n\n")

        for O in others:
            p = f"{c_name(O['path'])}_{O['op'].upper()}"
            f.write(f"// /{O['path']} ({O['op']}): not a Gemmini conv/matmul, run it separately\n")
            for i, (s, zp) in enumerate(O["ins"]):
                f.write(f"#define {p}_IN{i}_SCALE {s:.10e}f\n#define {p}_IN{i}_ZERO_POINT {zp}\n")
            if O["out"]:
                f.write(f"#define {p}_OUT_SCALE {O['out'][0]:.10e}f\n#define {p}_OUT_ZERO_POINT {O['out'][1]}\n")
            if "negative_slope" in O:
                f.write(f"#define {p}_NEGATIVE_SLOPE {O['negative_slope']}f\n")
            f.write("\n")

        for L in computes:
            wname = COMPAT_NAMES.get(L["path"]) or L["path"].replace(".", "_") + "_weights"
            bname = wname.replace("_weights_flat", "_bias").replace("_weights", "_bias")
            layout = "[KH*KW*IC][OC]" if L["kind"] == "conv" else "[OUT][IN]"
            w, b = L["w"], L["bias"]
            f.write(f"static elem_t {wname}[{w.shape[0]}][{w.shape[1]}] = {{  // {layout}\n{fmt_2d(w)}\n}};\n\n")
            f.write(f"static acc_t {bname}[{len(b)}] = {{\n{fmt_1d(b)}\n}};\n\n")

    for L in computes:
        print(f"{L['path']:16s} {L['kind']:6s} w{list(L['w'].shape)} acc_scale={L['acc_scale']:.6e} {L['act']}")
    for O in others:
        print(f"{O['path']:16s} {O['op']:20s} not a Gemmini conv/matmul: run it separately")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
