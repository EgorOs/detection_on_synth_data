CUSTOM_HOOKS = (
    dict(type='NumClassCheckHook'),
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0002, update_buffers=True, priority=49),
)
