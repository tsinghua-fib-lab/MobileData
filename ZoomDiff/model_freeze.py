def freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_fewshot_modules(model):
    # 1. 条件投影（ResidualBlock 内）
    for m in model.diffmodel.residual_layers:
        m.cond_static_projection.requires_grad_(True)
        m.cond_poi_projection.requires_grad_(True)
        m.side_projection.requires_grad_(True)

    # 2. embedding / side info
    model.embed_layer.requires_grad_(True)
    # for p in model.spatial_embedding:
    #     p.requires_grad = True

    # 3. 输出相关
    model.predicted_noise_projection.requires_grad_(True)
    model.noise_prior_projection.requires_grad_(True)
    model.infused_noise_projection.requires_grad_(True)

    # 4. 最后一层 ResidualBlock
    for block in model.diffmodel.residual_layers[-1:]:
        for p in block.parameters():
            p.requires_grad = True