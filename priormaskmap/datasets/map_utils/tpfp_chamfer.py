import numpy as np
from shapely.geometry import LineString
from shapely.strtree import STRtree
from shapely.geometry import CAP_STYLE, JOIN_STYLE
from scipy.spatial import distance


def custom_polyline_score(pred_lines, gt_lines, linewidth=1.0, metric="chamfer"):
    if metric == "iou":
        linewidth = 1.0
    positive_threshold = 1.0
    num_preds = len(pred_lines)
    num_gts = len(gt_lines)
    line_length = pred_lines.shape[1]
    pred_lines_shapely = [
        LineString(i).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre
        )
        for i in pred_lines
    ]
    gt_lines_shapely = [
        LineString(i).buffer(
            linewidth, cap_style=CAP_STYLE.flat, join_style=JOIN_STYLE.mitre
        )
        for i in gt_lines
    ]
    tree = STRtree(pred_lines_shapely)
    index_by_id = dict(((id(pt), i) for i, pt in enumerate(pred_lines_shapely)))
    if metric == "chamfer":
        iou_matrix = np.full((num_preds, num_gts), -100.0)
    elif metric == "iou":
        iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float64)
    for i, pline in enumerate(gt_lines_shapely):
        for o in tree.query(pline):
            if o.intersects(pline):
                pred_id = index_by_id[id(o)]
                if metric == "chamfer":
                    dist_mat = distance.cdist(
                        pred_lines[pred_id], gt_lines[i], "euclidean"
                    )
                    valid_ab = dist_mat.min(-1).mean()
                    valid_ba = dist_mat.min(-2).mean()
                    iou_matrix[pred_id, i] = -(valid_ba + valid_ab) / 2
                elif metric == "iou":
                    inter = o.intersection(pline).area
                    union = o.union(pline).area
                    iou_matrix[pred_id, i] = inter / union
    return iou_matrix
