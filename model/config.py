model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False,
        batch_augments=None,
    ),
    backbone=dict(
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.167,
        widen_factor=0.375,
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth',
        ),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[96, 192, 384],
        out_channels=96,
        num_csp_blocks=1,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=1,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        anchor_generator=dict(type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=False,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU', inplace=True),
    ),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=300,
    ),
)
work_dir = '/home/egor/Projects/detection_on_synth_data/src/mm_detection/mmdet_training_results/1695762346'
train_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='/home/egor/Projects/detection_on_synth_data/dataset/35',
        ann_file='/dataset/_35/train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0, max_cached_images=20, random_pop=False),
            dict(type='RandomResize', scale=(1280, 1280), ratio_range=(0.5, 2.0), keep_ratio=True),
            dict(type='RandomCrop', crop_size=(640, 640)),
            dict(type='YOLOXHSVRandomAug'),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
            dict(
                type='CachedMixUp',
                img_scale=(640, 640),
                ratio_range=(1.0, 1.0),
                max_cached_images=10,
                random_pop=False,
                pad_val=(114, 114, 114),
                prob=0.5,
            ),
            dict(type='PackDetInputs'),
        ],
        backend_args=None,
        metainfo=dict(classes=('stop_sign',)),
    ),
    pin_memory=True,
)
val_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='/home/egor/Projects/detection_on_synth_data/dataset/35',
        ann_file='/dataset/_35/val.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
        ],
        backend_args=None,
        metainfo=dict(classes=('stop_sign',)),
    ),
    pin_memory=False,
)
test_dataloader = dict(
    batch_size=12,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='CocoDataset',
        data_root='/home/egor/Projects/detection_on_synth_data/dataset/35',
        ann_file='/dataset/_35/test.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=[
            dict(type='LoadImageFromFile', backend_args=None),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
            dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')),
        ],
        backend_args=None,
        metainfo=dict(classes=('stop_sign',)),
    ),
    pin_memory=False,
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
auto_scale_lr = dict(enable=False, base_batch_size=16)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=0,
        end=100,
        T_max=33,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
val_evaluator = dict(
    type='CocoMetric',
    ann_file='/dataset/_35/val.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10),
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file='/dataset/_35/test.json',
    metric='bbox',
    format_only=False,
    backend_args=None,
    proposal_nums=(100, 1, 10),
)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=20),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=5, max_keep_ckpts=1, save_best='coco/bbox_mAP', rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)
custom_hooks = (
    {'type': 'NumClassCheckHook'},
    {'type': 'EMAHook', 'ema_type': 'ExpMomentumEMA', 'momentum': 0.0002, 'update_buffers': True, 'priority': 49},
)
data_preprocessor = None
load_from = None
launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_processor = None
log_level = 'INFO'
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend'), dict(type='ClearMLVisBackend', artifact_suffix=('.py', '.pth'))],
    name='visualizer',
)
default_scope = 'mmdet'
randomness = dict(seed=0)
experiment_name = 'experiment_1'
cfg = None
