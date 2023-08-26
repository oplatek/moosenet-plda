#!/bin/bash
set -eo pipefail

id=-1
ckpt=
split=dev
data_setup=voicemos_ood1_labeledonly
gpus=1

. utils/parse_options.sh
. utils/experiment.sh

experiment_id=$id  # use just small letters like a, aa, b, ab, etc.
if [ $experiment_id -lt 0 ] ; then printf "\nPlease set experiment_id to valid number\n\n"; exit 1 ; fi
# set -x

printf "Running experiment id=$experiment_id\n"

# In paper
#     % see ood.sh 30-39 or wandb voicemos_plda_train 3[0-9]_EMB
#     W2V\_main      & 2.657$\pm$0.399 & 0.710$\pm$0.040 \\ 
if (($experiment_id == 30)); then
  original_id=$experiment_id
  # experiments 30 31 32 33 34 35 36 37 38 39
  IB_1010_ckptdirs="$(ls -d exp/101[0-9]*/voicemos_main/*/checkpoints/)"
  if [[ $(echo $IB_1010_ckptdirs | wc -w) -ne 10 ]] ; then
    printf "There is not 10 experiments in IB_1010_ckptdirs. Exiting:\n\t${IB_1010_ckptdirs}\n"
    exit 1
  fi
  export NO_TAIL_NO_BLOCK=yes
  for d in $IB_1010_ckptdirs ; do
    moose 1 24G 16 " infer \
      $d \
      --ckpt_best_from_dir 1 \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_ood1_labeledonly \
      --time_augment_train 21 \
      --max_batch_duration 100 \
    "
    # fails with default gauss_noise 0.0 ie without it  
    #   numpy.linalg.LinAlgError: The leading minor of order 31 of B is not positive definite. The factorization of B could not be completed and no eigenvalues or eigenvectors were computed.
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath --add_gauss_noise 0.001 \
      "
    experiment_id=$(($experiment_id + 1))
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi

# In paper
#     % see ood.sh 40-49 or wandb voicemos_plda_train 4[0-9]_EMB
#     XLSR\_main    & 2.630$\pm$0.301 & 0.748$\pm$0.041 \\
if (($experiment_id == 40)); then
  original_id=$experiment_id
  # experiments 40 41 42 43 44 45 46 47 48 49
  XLSR_1020_ckptdirs="$(ls -d exp/102[0-9]*/voicemos_main/*/checkpoints/)"
  if [[ $(echo $XLSR_1020_ckptdirs | wc -w) -ne 10 ]] ; then
    printf "There is not 10 experiments in XLSR_1020_ckptdirs. Exiting:\n\t${XLSR_1020_ckptdirs}\n"
    exit 1
  fi
  export NO_TAIL_NO_BLOCK=yes
  for d in $XLSR_1020_ckptdirs ; do
    moose 1 24G 16 " infer \
      $d \
      --ckpt_best_from_dir 1 \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_ood1_labeledonly \
      --time_augment_train 21 \
      --max_batch_duration 100 \
    "
    # fails with default gauss_noise 0.0 ie without it  
    #   numpy.linalg.LinAlgError: The leading minor of order 31 of B is not positive definite. The factorization of B could not be completed and no eigenvalues or eigenvectors were computed.
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath --add_gauss_noise 0.001 \
      "
    experiment_id=$(($experiment_id + 1))
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi

# In paper
#     % see ood.sh OOD 50-59 or wandb voicemos_plda_train 5[0-9]_EMB
#     W2V+PLDA\_ood & \textbf{0.057}$\pm$0.009 & \textbf{0.955}$\pm$0.001 \\
if (($experiment_id == 50)); then
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for i in 50 51 52 53 54 55 56 57 58 59 ; do
    experiment_id="${i}"
    moose 1 24G 16 " infer \
      wav2vec_small \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_ood1_labeledonly \
      --time_augment_train 21 \
      --max_batch_duration 100 \
      --decoder_num_layers 0 \
      --decoder_hidden_dim 768 \
    "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath --add_gauss_noise 0.01 --n_bins 16 --plda_feat_dim 64
      "
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi



# In paper
#     % see ood.sh 100000-10009 or wandb voicemos_plda_train 1000[0-9]_EMB
#     W2V\_ood      & 0.263$\pm$0.128 & 0.955$\pm$0.013 \\ 
if (($experiment_id == 10000)); then
  original_id=$experiment_id
  IB_1010_ckptdirs="$(ls -d exp/101[0-9]*/voicemos_main/*/checkpoints/)"
  if [[ $(echo $IB_1010_ckptdirs | wc -w) -ne 10 ]] ; then
    printf "There is not 10 experiments in IB_1010_ckptdirs. Exiting:\n\t${IB_1010_ckptdirs}\n"
    exit 1
  fi
  export NO_TAIL_NO_BLOCK=yes
  for d in $IB_1010_ckptdirs ; do
    moose 1 48G 16 "train \
      --seed $experiment_id \
      --data_setup voicemos_ood1_labeledonly \
      --ckpt_best_from_dir 1 \
      --load_weights $d \
      --max_batch_duration 140 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      --warmup_steps 1000 \
      --learning_rate 0.0001 \
      "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    moose 1 24G 16 " infer \
      $logpath \
      --ckpt_best_from_log \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_ood1_labeledonly \
      --time_augment_train 21 \
      --max_batch_duration 100 \
    "
    # fails with default gauss_noise 0.0 ie without it  
    #   numpy.linalg.LinAlgError: The leading minor of order 31 of B is not positive definite. The factorization of B could not be completed and no eigenvalues or eigenvectors were computed.
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath --add_gauss_noise 0.001 \
      "
    experiment_id=$(($experiment_id + 1))
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi

# In paper
#     % see ood.sh 100010-10019 or wandb voicemos_plda_train 1001[0-9]_EMB
#     XLSR\_ood  & \textbf{0.058}$\pm$0.011 & 0.942$\pm$0.007 \\
if (($experiment_id == 10010)); then
  original_id=$experiment_id
  XLSR_1020_ckptdirs="$(ls -d exp/102[0-9]*/voicemos_main/*/checkpoints/)"
  if [[ $(echo $XLSR_1020_ckptdirs | wc -w) -ne 10 ]] ; then
    printf "There is not 10 experiments in XLSR_1020_ckptdirs. Exiting:\n\t${XLSR_1020_ckptdirs}\n"
    exit 1
  fi
  export NO_TAIL_NO_BLOCK=yes
  for d in $XLSR_1020_ckptdirs ; do
    moose 1 48G 16 "train \
      --seed $experiment_id \
      --data_setup voicemos_ood1_labeledonly \
      --ckpt_best_from_dir 1 \
      --load_weights $d \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      --warmup_steps 1000 \
      --learning_rate 0.0001 \
      "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    moose 1 24G 16 " infer \
      $logpath \
      --ckpt_best_from_log \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_ood1_labeledonly \
      --time_augment_train 21 \
      --max_batch_duration 100 \
    "
    # fails with default gauss_noise 0.0 ie without it  
    #   numpy.linalg.LinAlgError: The leading minor of order 31 of B is not positive definite. The factorization of B could not be completed and no eigenvalues or eigenvectors were computed.
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath --add_gauss_noise 0.001 \
      "
    experiment_id=$(($experiment_id + 1))
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi


# In paper
#     % see ood.sh 10070-10079
#     W2V\_solo-ood & 0.265$\pm$0.144 & 0.927$\pm$0.023 \\
if (($experiment_id == 10070)); then
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for experiment_id in 10070 10071 10072 10073 10074 10075 10076 10077 10078 10079 ; do
    moose 1 48G 16 "train \
      --seed $experiment_id \
      --data_setup voicemos_ood1_labeledonly \
      --max_batch_duration 140 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    moose 1 24G 16 " infer \
      $logpath \
      --ckpt_best_from_log \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_ood1_labeledonly \
      --time_augment_train 21 \
      --max_batch_duration 100 \
    "
    # fails with default gauss_noise 0.0 ie without it  
    #   numpy.linalg.LinAlgError: The leading minor of order 31 of B is not positive definite. The factorization of B could not be completed and no eigenvalues or eigenvectors were computed.
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath --add_gauss_noise 0.001 \
      "
    experiment_id=$(($experiment_id + 1))
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi
