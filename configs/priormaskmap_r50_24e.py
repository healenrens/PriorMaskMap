point_cloud_range = [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0]

map_classes = ["divider", "ped_crossing", "boundary"]

data_root = "data/nuscenes/"

train_pipeline = [
    {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
    {"type": "PriorRandomScaleImageMultiViewImage", "scales": [0.5]},
    {"type": "PhotoMetricDistortionMultiViewImage"},
    {
        "type": "PriorNormalizeMultiviewImage",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "to_rgb": True,
    },
    {
        "type": "LoadPointsFromFile",
        "coord_type": "LIDAR",
        "load_dim": 5,
        "use_dim": 5,
        "file_client_args": {"backend": "disk"},
    },
    {
        "type": "PriorCustomPointToMultiViewDepth",
        "downsample": 1,
        "grid_config": {
            "x": [-30.0, -30.0, 0.15],
            "y": [-15.0, -15.0, 0.15],
            "z": [-10, 10, 20],
            "depth": [1.0, 35.0, 0.5],
        },
    },
    {"type": "PriorPadMultiViewImageDepth", "size_divisor": 32},
    {
        "type": "DefaultFormatBundle3D",
        "with_gt": False,
        "with_label": False,
        "class_names": ["divider", "ped_crossing", "boundary"],
    },
    {"type": "PriorCustomCollect3D", "keys": ["img", "gt_depth"]},
]

test_pipeline = [
    {"type": "LoadMultiViewImageFromFiles", "to_float32": True},
    {"type": "PriorRandomScaleImageMultiViewImage", "scales": [0.5]},
    {
        "type": "PriorNormalizeMultiviewImage",
        "mean": [123.675, 116.28, 103.53],
        "std": [58.395, 57.12, 57.375],
        "to_rgb": True,
    },
    {
        "type": "MultiScaleFlipAug3D",
        "img_scale": (1600, 900),
        "pts_scale_ratio": 1,
        "flip": False,
        "transforms": [
            {"type": "PriorPadMultiViewImage", "size_divisor": 32},
            {
                "type": "DefaultFormatBundle3D",
                "with_gt": False,
                "with_label": False,
                "class_names": ["divider", "ped_crossing", "boundary"],
            },
            {"type": "PriorCustomCollect3D", "keys": ["img"]},
        ],
    },
]

data = {
    "samples_per_gpu": 4,
    "workers_per_gpu": 4,
    "train": {
        "type": "PriorMapDataset",
        "data_root": "data/nuscenes/",
        "ann_file": "data/nuscenes/nuscenes_map_infos_temporal_train.pkl",
        "pipeline": None,
        "classes": [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
        "modality": {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        },
        "test_mode": False,
        "box_type_3d": "LiDAR",
        "aux_seg": {
            "use_aux_seg": True,
            "bev_seg": True,
            "pv_seg": True,
            "seg_classes": 1,
            "feat_down_sample": 32,
            "pv_thickness": 1,
        },
        "use_valid_flag": True,
        "bev_size": (200, 100),
        "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        "fixed_ptsnum_per_line": 20,
        "eval_use_same_gt_sample_num_flag": True,
        "padding_value": -10000,
        "map_classes": ["divider", "ped_crossing", "boundary"],
        "queue_length": 1,
    },
    "val": {
        "type": "PriorMapDataset",
        "ann_file": "data/nuscenes/nuscenes_map_infos_temporal_val.pkl",
        "pipeline": None,
        "classes": [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
        "modality": {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        },
        "test_mode": True,
        "box_type_3d": "LiDAR",
        "data_root": "data/nuscenes/",
        "map_ann_file": "data/nuscenes/nuscenes_map_anns_val.json",
        "bev_size": (200, 100),
        "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        "fixed_ptsnum_per_line": 20,
        "eval_use_same_gt_sample_num_flag": True,
        "padding_value": -10000,
        "map_classes": ["divider", "ped_crossing", "boundary"],
        "samples_per_gpu": 1,
    },
    "test": {
        "type": "PriorMapDataset",
        "data_root": "data/nuscenes/",
        "ann_file": "data/nuscenes/nuscenes_map_infos_temporal_val.pkl",
        "pipeline": None,
        "classes": [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ],
        "modality": {
            "use_lidar": False,
            "use_camera": True,
            "use_radar": False,
            "use_map": False,
            "use_external": True,
        },
        "test_mode": True,
        "box_type_3d": "LiDAR",
        "map_ann_file": "data/nuscenes/nuscenes_map_anns_val.json",
        "bev_size": (200, 100),
        "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
        "fixed_ptsnum_per_line": 20,
        "eval_use_same_gt_sample_num_flag": True,
        "padding_value": -10000,
        "map_classes": ["divider", "ped_crossing", "boundary"],
    },
    "shuffler_sampler": {"type": "PriorDistributedGroupSampler"},
    "nonshuffler_sampler": {"type": "PriorDistributedSampler"},
}

map_cfg = {
    "load_map_path": "data/priors/correct.pkl",
    "map_modification_data_path": [
        "data/priors/mixed_train.pkl",
        "data/priors/mixed_val.pkl",
    ],
    "change_type": {
        "train": [["mix4cats", "train", "v1"]],
        "val": [["mix4cats", "test", "v1"]],
    },
    "patch_size": 10,
}

model = {
    "type": "PriorMaskMap",
    "use_grid_mask": True,
    "video_test_mode": False,
    "global_map_cfg": None,
    "pretrained": {"img": "checkpoints/resnet50.pth"},
    "img_backbone": {
        "type": "ResNet",
        "depth": 50,
        "num_stages": 4,
        "out_indices": (3,),
        "frozen_stages": 1,
        "norm_cfg": {"type": "BN", "requires_grad": False},
        "norm_eval": True,
        "style": "pytorch",
    },
    "img_neck": {
        "type": "FPN",
        "in_channels": [2048],
        "out_channels": 256,
        "start_level": 0,
        "add_extra_convs": "on_output",
        "num_outs": 1,
        "relu_before_extra_convs": True,
    },
    "pts_bbox_head": {
        "type": "PriorMaskMapHead",
        "bev_h": 200,
        "bev_w": 100,
        "num_query": 900,
        "num_vec_one2one": 50,
        "num_vec_one2many": 300,
        "k_one2many": 6,
        "num_pts_per_vec": 20,
        "num_pts_per_gt_vec": 20,
        "dir_interval": 1,
        "query_embed_type": "map_pts",
        "map_query_scale": 2,
        "transform_method": "minmax",
        "gt_shift_pts_pattern": "v2",
        "num_classes": 3,
        "in_channels": 256,
        "sync_cls_avg_factor": True,
        "with_box_refine": True,
        "as_two_stage": False,
        "code_size": 2,
        "code_weights": [1.0, 1.0, 1.0, 1.0],
        "aux_seg": {
            "use_aux_seg": True,
            "bev_seg": True,
            "pv_seg": True,
            "seg_classes": 1,
            "feat_down_sample": 32,
            "pv_thickness": 1,
        },
        "enable_error_detection": True,
        "error_detection_loss_weight": 1.0,
        "vector_embed_dim": 256,
        "vector_num_heads": 4,
        "transformer": {
            "type": "PriorFusionTransformer",
            "query_pos_embedding": "map_pts",
            "rotate_prev_bev": True,
            "use_shift": True,
            "use_can_bus": True,
            "embed_dims": 256,
            "enable_expert_fusion": True,
            "vector_query_weight": 0.5,
            "fusion_type": "uncertainty_guided_attention",
            "map_encoder": {
                "type": "PriorAttentionEncoder",
                "num_layers": 1,
                "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
                "num_points_in_pillar": 4,
                "return_intermediate": False,
                "transformerlayers": {
                    "type": "PriorAttentionLayer",
                    "attn_cfgs": [
                        {
                            "type": "PriorTemporalSelfAttention",
                            "embed_dims": 256,
                            "num_levels": 1,
                        },
                        {
                            "type": "PriorSpatialCrossAttention",
                            "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
                            "deformable_attention": {
                                "type": "PriorMSDeformableAttention3D",
                                "embed_dims": 256,
                                "num_points": 8,
                                "num_levels": 1,
                            },
                            "embed_dims": 256,
                        },
                    ],
                    "feedforward_channels": 512,
                    "ffn_dropout": 0.1,
                    "operation_order": (
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                },
            },
            "encoder": {
                "type": "PriorLSSTransform",
                "in_channels": 256,
                "out_channels": 256,
                "feat_down_sample": 32,
                "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
                "voxel_size": [0.15, 0.15, 20.0],
                "dbound": [1.0, 35.0, 0.5],
                "downsample": 2,
                "loss_depth_weight": 3.0,
                "depthnet_cfg": {
                    "use_dcn": False,
                    "with_cp": False,
                    "aspp_mid_channels": 96,
                },
                "grid_config": {
                    "x": [-30.0, -30.0, 0.15],
                    "y": [-15.0, -15.0, 0.15],
                    "z": [-10, 10, 20],
                    "depth": [1.0, 35.0, 0.5],
                },
            },
            "decoder": {
                "type": "PolylineDecoder",
                "num_layers": 6,
                "return_intermediate": True,
                "transformerlayers": {
                    "type": "PriorDecoupledDetrTransformerDecoderLayer",
                    "num_vec": 50,
                    "num_pts_per_vec": 20,
                    "attn_cfgs": [
                        {
                            "type": "MultiheadAttention",
                            "embed_dims": 256,
                            "num_heads": 8,
                            "dropout": 0.1,
                        },
                        {
                            "type": "MultiheadAttention",
                            "embed_dims": 256,
                            "num_heads": 8,
                            "dropout": 0.1,
                        },
                        {
                            "type": "PriorCustomMSDeformableAttention",
                            "embed_dims": 256,
                            "num_levels": 1,
                        },
                    ],
                    "feedforward_channels": 512,
                    "ffn_dropout": 0.1,
                    "operation_order": (
                        "self_attn",
                        "norm",
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                },
            },
        },
        "bbox_coder": {
            "type": "PolylineCoder",
            "post_center_range": [-20, -35, -20, -35, 20, 35, 20, 35],
            "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
            "max_num": 50,
            "voxel_size": [0.15, 0.15, 20.0],
            "num_classes": 3,
        },
        "positional_encoding": {
            "type": "LearnedPositionalEncoding",
            "num_feats": 128,
            "row_num_embed": 200,
            "col_num_embed": 100,
        },
        "loss_cls": {
            "type": "FocalLoss",
            "use_sigmoid": True,
            "gamma": 2.0,
            "alpha": 0.25,
            "loss_weight": 2.0,
        },
        "loss_bbox": {"type": "L1Loss", "loss_weight": 0.0},
        "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
        "loss_pts": {"type": "PriorPtsL1Loss", "loss_weight": 5.0},
        "loss_dir": {"type": "PriorPtsDirCosLoss", "loss_weight": 0.005},
        "loss_seg": {"type": "PriorSimpleLoss", "pos_weight": 4.0, "loss_weight": 1.0},
        "loss_pv_seg": {
            "type": "PriorSimpleLoss",
            "pos_weight": 1.0,
            "loss_weight": 2.0,
        },
    },
    "train_cfg": {
        "pts": {
            "grid_size": [512, 512, 1],
            "voxel_size": [0.15, 0.15, 20.0],
            "point_cloud_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
            "out_size_factor": 4,
            "assigner": {
                "type": "PolylineAssigner",
                "cls_cost": {"type": "FocalLossCost", "weight": 2.0},
                "reg_cost": {"type": "BBoxL1Cost", "weight": 0.0, "box_format": "xywh"},
                "iou_cost": {"type": "IoUCost", "iou_mode": "giou", "weight": 0.0},
                "pts_cost": {"type": "PriorOrderedPtsL1Cost", "weight": 5},
                "pc_range": [-15.0, -30.0, -10.0, 15.0, 30.0, 10.0],
            },
        }
    },
}

optimizer = {
    "type": "AdamW",
    "lr": 0.0006,
    "paramwise_cfg": {
        "custom_keys": {
            "img_backbone": {"lr_mult": 0.1},
            "error_detection_unet": {"lr_mult": 2.0},
            "uncertainty_gate": {"lr_mult": 1.0},
        }
    },
    "weight_decay": 0.01,
}

optimizer_config = {"grad_clip": {"max_norm": 35, "norm_type": 2}}

lr_config = {
    "policy": "CosineAnnealing",
    "warmup": "linear",
    "warmup_iters": 500,
    "warmup_ratio": 0.3333333333333333,
    "min_lr_ratio": 0.001,
}

total_epochs = 24

runner = {"type": "EpochBasedRunner", "max_epochs": 24}

custom_hooks = [
    {"type": "PriorSetEpochInfoHook", "priority": "VERY_HIGH", "interval": 1}
]

log_config = {
    "interval": 10,
    "hooks": [{"type": "TextLoggerHook"}, {"type": "TensorboardLoggerHook"}],
}

checkpoint_config = {"interval": 1, "max_keep_ckpts": 4}

fp16 = {"loss_scale": 512.0}

find_unused_parameters = True

dist_params = {"backend": "nccl"}

log_level = "INFO"

workflow = [("train", 1)]

evaluation = {
    "interval": 2,
    "metric": "chamfer",
    "save_best": "NuscMap_chamfer/mAP",
    "rule": "greater",
}

work_dir = "work_dirs/priormaskmap_r50_24e"

load_from = None

resume_from = None

model["global_map_cfg"] = map_cfg

data["train"]["pipeline"] = train_pipeline

data["val"]["pipeline"] = test_pipeline

data["test"]["pipeline"] = test_pipeline
