
python scripts/gr00t_finetune.py \
    --dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --output-dir /home/innovation-hacking/luebbet/models/eval_09_21/AbsJoint-AbsTCP-State_DeltaJoint-Action_tokenized_right \
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
    --save-steps 5000 \
    --train-modality-tokenizer \


python scripts/gr00t_finetune.py \
    --dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --output-dir /home/innovation-hacking/luebbet/models/eval_09_21/AbsJoint-AbsTCP-State_DeltaJoint-Action_deltaaas \
    --max-steps 35000 \
    --data-config "tng_ur5_AbsAndDeltaJointState_DeltaJointAction_2Cams" \
    --embodiment-tag new_embodiment \
    --video-backend torchvision_av \
    --batch-size 16 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --report-to tensorboard \
    --save-steps 5000 \
    --train-modality-tokenizer \


python scripts/gr00t_finetune.py \
    --dataset-path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --output-dir /home/innovation-hacking/luebbet/models/eval_09_21/AbsJoint-AbsTCP-State_DeltaJoint-Action_tokenized_right_bs8 \
    --max-steps 35000 \
    --data-config "tng_ur5_AbsJointAndAbsTCPState_DeltaJointAction_2Cams" \
    --embodiment-tag new_embodiment \
    --video-backend torchvision_av \
    --batch-size 8 \
    --no-tune-llm \
    --no-tune-visual \
    --tune-projector \
    --tune-diffusion-model \
    --report-to tensorboard \
    --save-steps 5000 \
    --train-modality-tokenizer \
