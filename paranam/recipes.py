"""Named starting recipes; user edits remain separately persisted per capture type."""

PRESET_SOURCE = "https://github.com/phillipmself/neural-amp-modeler-parametric/tree/a3684411076be69dab53505df14922487697f2cf"


def training_preset(name):
    if name not in ("HyperA2", "HyperWaveNet"):
        raise ValueError(f"Unknown training preset: {name}")
    hyper_a2 = name == "HyperA2"
    return dict(
        hypernetwork_recipe=name,
        optimizer="Adam" if hyper_a2 else "AdamW",
        learning_rate=0.004 if hyper_a2 else 0.002,
        hypernetwork_learning_rate=0.0004 if hyper_a2 else 0.002,
        lr_decay=0.994,
        ny=8192,
        fit_mrstft=True,
        mrstft_weight=0.0005,
        weight_decay=3.17e-7,
        gradient_clipping=True,
        gradient_clip_val=1.0,
        check_nonfinite_gradients=True,
    )


def preset_batch_size(name):
    return 16 if name == "HyperA2" else 32
