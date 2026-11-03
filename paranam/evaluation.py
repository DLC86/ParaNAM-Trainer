"""Held-out audio evaluation for captured training control settings."""

from copy import deepcopy

PROTOCOL = "nam-v3-held-out-v1"


def reserve_validation_audio(data):
    """Use the official NAM v3 training split and final nine-second validation.

    Standard and TTS v11 inputs retain this layout. Keep delay, paths and
    controls per capture; validation recordings remain a separate data split.
    """
    evaluation = []
    for entry in data["train"]:
        held_out = deepcopy(entry)
        for row in (entry, held_out):
            for key in ("start", "stop", "start_samples", "stop_samples"):
                row.pop(key, None)
        entry.update(start_seconds=10.0, stop_seconds=-9.0)
        held_out.update(start_seconds=-9.0, stop_seconds=None, ny=None)
        evaluation.append(held_out)
    data["train_evaluation"] = evaluation


class TrainEvaluator:
    """Evaluate frozen end-of-epoch weights; average ESR equally over captures."""

    def __init__(self, data, model_config, net):
        from nam.data import ConcatDataset, Split, init_dataset

        config = deepcopy(data)
        config["validation"] = config.pop("train_evaluation")
        for capture in config["validation"]:
            capture["ny"] = None  # One complete validation clip, as in official NAM.
        config.setdefault("common", {})["nx"] = net.receptive_field
        if config.get("type") == "parametric":
            from nam.models.parametric import data_config_from_model

            config = data_config_from_model(config, model_config)
        self.dataset = init_dataset(config, Split.VALIDATION)
        self.captures = (
            self.dataset.datasets
            if isinstance(self.dataset, ConcatDataset)
            else (self.dataset,)
        )

    def compute(self, net, device, check_action=lambda: None):
        import torch
        from nam.models.losses import esr

        modes = [(module, module.training) for module in net.modules()]
        totals = {"lite": [], "full": []}
        cuda_devices = [device.index or 0] if device.type == "cuda" else []
        try:
            net.eval()
            with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
                for capture in self.captures:
                    check_action()
                    batch = tuple(t.unsqueeze(0).to(device) for t in capture[0])
                    target = batch[-1]
                    predictions = (
                        net.forward_sizes(*batch[:-1], pad_start=False)
                        if hasattr(net, "forward_sizes")
                        else net(*batch[:-1], pad_start=False)
                    )
                    for slot, size in enumerate(("lite", "full")):
                        totals[size].append(esr(predictions[:, slot, :], target).item())
        finally:
            for module, mode in modes:
                module.training = mode
        return {size: sum(values) / len(values) for size, values in totals.items()}

    def close(self):
        self.dataset.teardown()
