import numpy as np
from priormaskmap.datasets.map_utils.tpfp_chamfer import custom_polyline_score


def custom_tpfp_gen(gen_lines, gt_lines, threshold=0.5, metric="chamfer", code_size=2):
    if metric == "chamfer":
        if threshold > 0:
            threshold = -threshold
    num_gens = gen_lines.shape[0]
    num_gts = gt_lines.shape[0]
    tp = np.zeros(num_gens, dtype=np.float32)
    fp = np.zeros(num_gens, dtype=np.float32)
    if num_gts == 0:
        fp[...] = 1
        return (tp, fp)
    if num_gens == 0:
        return (tp, fp)
    gen_scores = gen_lines[:, -1]
    matrix = custom_polyline_score(
        gen_lines[:, :-1].reshape(num_gens, -1, code_size),
        gt_lines.reshape(num_gts, -1, code_size),
        linewidth=2.0,
        metric=metric,
    )
    matrix_max = matrix.max(axis=1)
    matrix_argmax = matrix.argmax(axis=1)
    sort_inds = np.argsort(-gen_scores)
    gt_covered = np.zeros(num_gts, dtype=bool)
    for i in sort_inds:
        if matrix_max[i] >= threshold:
            matched_gt = matrix_argmax[i]
            if not gt_covered[matched_gt]:
                gt_covered[matched_gt] = True
                tp[i] = 1
            else:
                fp[i] = 1
        else:
            fp[i] = 1
    return (tp, fp)
