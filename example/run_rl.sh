python run_behavior_clone.py \
--env_name 'gym_flatworld:Flatworld-v0' \
--num_parallel 4 \
--cuda 0 \
--record \
--expert_dir "/home/hirobuchi.ryota/rl_lab/GAILs/data" \
--expert_fname "expert_gym_flatworld:Flatworld-v0_10_s42.pkl" \
--log "bc_models"
# --max_epis 100 \
