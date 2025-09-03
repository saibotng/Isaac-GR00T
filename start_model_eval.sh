python scripts/eval_policy.py \
    --plot \
    --dataset_path /home/innovation-hacking/luebbet/dev/important_datasets/pick_and_place_increased_obs_limited_rotation_corrupted \
    --model_path /home/innovation-hacking/luebbet/models/eval_08_14/diff_small_obs_delta_actions/checkpoint-13000/ \
    --embodiment_tag new_embodiment \
    --data_config "tng_ur5_schunk_2_cams_sim_delta_actions" \
    --video_backend torchvision_av \
    --modality_keys delta_robot_arm delta_gripper \
    --save_plot_path /home/innovation-hacking/luebbet/models/eval_08_14/diff_small_obs_delta_actions/checkpoint-13000/plots_unseen \
    --trajs 10