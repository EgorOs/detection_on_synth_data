default_scope = 'mmdet'
max_epochs = 100
base_lr = 0.003
val_interval = 5
log_level = 'INFO'
load_from = None
resume = False

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(
        type='LoggerHook',
        interval=20,
    ),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=val_interval,
        max_keep_ckpts=1,  # Minimize ClearML storage consumption.
        save_best='coco/bbox_mAP',  # Best checkpoint is required to build model archive for torchserve.
        rule='greater',
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='ClearMLVisBackend', artifact_suffix=('.py', '.pth')),  # ClearML logging and artifact storage.
    ],
    name='visualizer',
)
log_processor = dict(type='LogProcessor', window_size=20, by_epoch=True)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=val_interval)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0002,
        begin=0,
        end=max_epochs,
        T_max=max_epochs // 3,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.004, weight_decay=0.05),
    paramwise_cfg=dict(norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True),
)
auto_scale_lr = dict(enable=False, base_batch_size=16)
