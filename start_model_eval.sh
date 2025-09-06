python scripts/eval_policy.py \
    --plot \
    --dataset_path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --model_path /home/innovation-hacking/luebbet/models/eval_09_16/AbsJoint-State_DeltaJoint-Action/checkpoint-35000 \
    --embodiment_tag new_embodiment \
    --data_config "tng_ur5_AbsJointState_DeltaJointAction_2Cams" \
    --video_backend torchvision_av \
    --modality_keys delta_robot_arm delta_gripper \
    --save_plot_path /home/innovation-hacking/luebbet/models/eval_09_16/AbsJoint-State_DeltaJoint-Action/checkpoint-35000/plots \
    --trajs 1