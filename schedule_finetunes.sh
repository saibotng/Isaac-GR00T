python scripts/gr00t_finetune.py \
    --train-dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --validation-dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/validation/ \
    --eval_steps 0 \
    --output-dir /home/innovation-hacking/luebbet/models/eval_09_16/AbsJoint-AbsTCP-State_DeltaJoint-Action \
    --max-steps 35000 \
    --data-config "tng_ur5_AbsJointAndAbsTCPState_DeltaJointAction_2Cams" \
    --embodiment-tag new_embodiment \
    --video-backend torchvision_av \
    --batch-size 16 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --report-to tensorboard \
    --save-steps 1000 \
    --train-modality-tokenizer \


python scripts/gr00t_finetune.py \
    --train-dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --validation-dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/validation/ \
    --eval_steps 0 \
    --output-dir /home/innovation-hacking/luebbet/models/eval_09_16/AbsJoint-State_DeltaJoint-Action \
    --max-steps 35000 \
    --data-config "tng_ur5_AbsJointState_DeltaJointAction_2Cams" \
    --embodiment-tag new_embodiment \
    --video-backend torchvision_av \
    --batch-size 16 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --report-to tensorboard \
    --save-steps 1000 \
    --train-modality-tokenizer \


python scripts/gr00t_finetune.py \
    --train-dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --validation-dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/validation/ \
    --eval_steps 1000 \
    --output-dir /home/innovation-hacking/luebbet/models/eval_09_16/AbsJoint-AbsTCP-State_DeltaJoint-Action_Validation \
    --max-steps 35000 \
    --data-config "tng_ur5_AbsJointAndAbsTCPState_DeltaJointAction_2Cams" \
    --embodiment-tag new_embodiment \
    --video-backend torchvision_av \
    --batch-size 16 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --report-to tensorboard \
    --save-steps 1000 \
    --train-modality-tokenizer \
    --eval-batch-size 8 \