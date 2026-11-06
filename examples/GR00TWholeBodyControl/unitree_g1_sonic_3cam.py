# 3-camera variant of the pre-registered `unitree_g1_sonic` modality config:
# video = ego_view + left_wrist + right_wrist (second G1: head D405 + 2 wrist D405s,
# recorded with --record-wrist-cameras). State/action/language are unchanged, so the
# embodiment tag stays `unitree_g1_sonic`.
#
# Train:  launch_finetune.py ... --embodiment-tag UNITREE_G1_SONIC \
#           --modality-config-path examples/GR00TWholeBodyControl/unitree_g1_sonic_3cam.py
# (do NOT also pass gr00t/configs/data/embodiment_configs.py — this file replaces that flag)
#
# The 3 video keys are saved into the checkpoint's experiment_cfg, so serving needs no
# extra flag here; the SONIC inference client (run_vla_inference.py) must send all three
# views under these exact names.

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.types import ModalityConfig

MODALITY_CONFIGS["unitree_g1_sonic"]["video"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=["ego_view", "left_wrist", "right_wrist"],
)
print("[unitree_g1_sonic_3cam] video keys:", MODALITY_CONFIGS["unitree_g1_sonic"]["video"].modality_keys)
