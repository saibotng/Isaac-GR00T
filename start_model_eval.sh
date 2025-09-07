python scripts/eval_policy.py \
    --plot \
    --dataset_path /home/innovation-hacking/luebbet/dev/important_datasets/experiments/01_kinematics/cal3_rand150_dim0-2_moving_target/ \
    --model_path /home/innovation-hacking/luebbet/models/eval_09_21/AbsJoint-AbsTCP-State_DeltaJoint-Action_no_embedds/checkpoint-50000 \
    --embodiment_tag new_embodiment \
    --data_config "tng_ur5_AbsJointAndAbsTCPState_DeltaJointAction_2Cams" \
    --video_backend torchvision_av \
    --modality_keys delta_robot_arm delta_gripper \
    --save_plot_path /home/innovation-hacking/luebbet/models/eval_09_21/AbsJoint-AbsTCP-State_DeltaJoint-Action_no_embedds/checkpoint-50000/plot \
    --trajs 1