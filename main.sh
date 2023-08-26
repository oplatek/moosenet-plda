#!/bin/bash
set -eo pipefail
# set -x

id=-1
. utils/parse_options.sh
. utils/experiment.sh

experiment_id=$id
if [ $experiment_id -lt 0 ] ; then printf "\nPlease set experiment_id to valid number\n\n"; exit 1 ; fi

# In paper
#     % Experiments 1040-1049
#     \textit{W2V\_main w/o STOI} & 0.140$\pm$0.033 & 0.922$\pm$0.007  \\
if (($experiment_id == 01040)); then
  : '
  See 104
  '
  moose 1 24G 9 "train \
    --max_batch_duration 40 \
    --dec_betas mos_final,1.0,var_mos_final,0.0,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
    --fairseq_model_path ../mos-finetune-ssl/fairseq-models/xlsr_53_56k.pt \
    "
    # --n_ckpts 0 \
fi

# In paper
#     % Experiments 1000-1009:
#     W2V\_main & 0.142$\pm$0.032 & \textbf{0.923}$\pm$0.006 \\
if (($experiment_id == 1000)); then
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  improved baseline (IB)
  '
  for experiment_id in 1000 1001 1002 1003 1004 1005 1006 1007 1008 1009 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi


# In paper
#     % Experiments 1010-1019
#     % TODO full 10 runs - 2 runs were missing due to cluster failure
#     \textit{W2V\_main 50\% train} & \textbf{0.150}$\pm$0.044 & \textbf{0.924}$\pm$0.006 \\
if (($experiment_id == 1010)); then
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  improved baseline (IB) larger batch size
  '
  for experiment_id in 1010 1011 1012 1013 1014 1015 1016 1017 1018 1019 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --max_batch_duration 140 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi


# In paper
#     % Experiments 1020-1029
#     XSLR\_main & 0.117$\pm$0.035 & 0.929$\pm$0.007 \\
if (($experiment_id == 1020)); then
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  XSLR
  '
  # for experiment_id in 1020 1021 1022 1023 1024 1025 1026 1027 1028 1029 ; do
  for experiment_id in 1022 1023 1026 1029 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --fairseq_model_path ../mos-finetune-ssl/fairseq-models/xlsr_53_56k.pt \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi

#  In paper
#     % Experiments 1050-1059
#     \textit{W2V\_main w/o contrast} & 0.149$\pm$0.033 & 0.922$\pm$0.007  \\
if (($experiment_id == 1050)); then 
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  IB w/o contrast
  '
  for experiment_id in 1050 1051 1052 1053 1054 1055 1056 1057 1058 1059 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.0 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi

# In paper
#     % Experiments 1060-1069
#     \textit{W2V\_main w/o augmnt.} & \textbf{0.137}$\pm$0.047 & 0.922$\pm$0.005 \\
if (($experiment_id == 1060)); then
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  improved baseline (IB) without_augmentation
  '
  for experiment_id in 1060 1061 1062 1063 1064 1065 1066 1067 1068 1069 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --without_augmentation \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi

if (($experiment_id == 1070)); then
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  improved baseline (IB) + listener modeling
  '
  for experiment_id in 1070 1071 1072 1073 1074 1075 1076 1077 1078 1079 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy random_avg \
      --listener_emb_size 32 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi

if (($experiment_id == 1090)); then
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  improved baseline (IB) + listener modeling
  '
  for experiment_id in 1090 1091 1092 1093 1094 1095 1096 1097 1098 1099 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy random_avg \
      --listener_emb_size 128 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi


# In paper
#     % Experiments 1080-1089
#     % W2V\_main$_{Gauss->logCosh}$ & 0.1594$\pm$0.035 &  0.922$\pm$0.0064  \\
#     \textit{W2V\_main\_logCosh/Gauss} & 0.159$\pm$0.035 &  0.922$\pm$0.006  \\
if (($experiment_id == 1080)); then
  : 'MooseNet: metrics for speech applications with PLDA backend
  table tab:ssl_baselines
  improved baseline (IB) - gaussloss + logCoshloss
  '
  for experiment_id in 1080 1081 1082 1083 1084 1085 1086 1087 1088 1089 ; do
    NO_TAIL_NO_BLOCK=on moose 1 48G 16 "train \
      --seed $experiment_id \
      --max_batch_duration 40 \
      --dec_betas mos_final,0.1,var_mos_final,0.0,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      "
  done
fi


if (($experiment_id == 1100)); then
  # See also ood.sh experiment_id == 50
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for i in 1100 1101 1102 1103 1104 1105 1106 1107 1108 1109 ; do
    experiment_id="${i}"
    moose 1 24G 16 " infer \
      wav2vec_small \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_main1 \
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
#     % train.sh 1110
#     W2V+PLDA\_main & 0.167$\pm$0.000 & \textbf{0.867}$\pm$0.000 \\
if (($experiment_id == 1110)); then
  # See also ood.sh experiment_id == 50
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for i in 1110 1111 1112 1113 1114 1115 1116 1117 1118 1119 ; do
    experiment_id="${i}"
    moose 1 24G 16 " infer \
      wav2vec_small \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_main1 \
      --time_augment_train 21 \
      --max_batch_duration 100 \
      --decoder_num_layers 0 \
      --decoder_hidden_dim 768 \
    "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath
      "
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi

if (($experiment_id == 1120)); then
  # See also ood.sh experiment_id == 50
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for i in 1120 1121 1122 1123 1124 1125 1126 1127 1128 1129 ; do
    experiment_id="${i}"
    moose 1 24G 16 " infer \
      xlsr \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_main1 \
      --time_augment_train 21 \
      --max_batch_duration 100 \
      --decoder_num_layers 0 \
      --decoder_hidden_dim 1024 \
    "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    cpu "\
      moosenet -v train-plda -e $experiment_id --seed $experiment_id --emb_dir_from_log $logpath
      "
  done
  unset NO_TAIL_NO_BLOCK
  experiment_id=$original_id
fi


if (($experiment_id == 1130)); then
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for experiment_id in 1130 1131 1132 1133 1134 1135 1136 1137 1138 1139 ; do
    moose 1 48G 16 "train \
      --seed $experiment_id \
      --data_setup voicemos_main1 \
      --max_batch_duration 140 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      --train_ratio 0.5 \
      "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    moose 1 24G 16 " infer \
      $logpath \
      --ckpt_best_from_log \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_main1 \
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

if (($experiment_id == 1140)); then
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  # for experiment_id in 1142 1144 1145 1146 ; do
  # for experiment_id in 1140 1141 1142 1143 1144 1145 1146 1147 1148 1149 ; do
  for experiment_id in 1140 1141 1143 1147 1148 1149 ; do
    moose 1 48G 16 "train \
      --seed $experiment_id \
      --data_setup voicemos_main1 \
      --max_batch_duration 140 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      --train_ratio 0.02734217933 \
      "
    # --train_ratio 0.02734217933 corresponds to 136 utterances
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    moose 1 24G 16 " infer \
      $logpath \
      --ckpt_best_from_log \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_main1 \
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


if (($experiment_id == 1150)); then
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for experiment_id in 1150 1151 1152 1153 1154 1155 1156 1157 1158 1159 ; do
    moose 1 48G 16 "train \
      --seed $experiment_id \
      --data_setup voicemos_main1 \
      --max_batch_duration 140 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      --train_ratio 0.10 \
      # corresponds to 136 utterances
      "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    moose 1 24G 16 " infer \
      $logpath \
      --ckpt_best_from_log \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_main1 \
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
#     % 1160-1169
#     \textit{W2V\_main 5\% train} & 0.307$\pm$0.176 & 0.884$\pm$0.006 \\
if (($experiment_id == 1160)); then
  original_id=$experiment_id
  export NO_TAIL_NO_BLOCK=yes
  for experiment_id in 1160 1161 1162 1163 1164 1165 1166 1167 1168 1169 ; do
    moose 1 48G 16 "train \
      --seed $experiment_id \
      --data_setup voicemos_main1 \
      --max_batch_duration 140 \
      --dec_betas mos_final,0.0,var_mos_final,0.1,snr,0.0,stoi,0.1,noise_label,0.0,consist_mos,0.0,mcd,0.0,contrast_mos,0.1 \
      --score_strategy mean \
      --listener_emb_size 0 \
      --decoder_hidden_dim 32 \
      --decoder_dropout 0.3 \
      --train_ratio 0.05 \
      "
    SLURM_DEPENDENCY_MOOSE="afterok:$job_id"
    moose 1 24G 16 " infer \
      $logpath \
      --ckpt_best_from_log \
      --seed $experiment_id \
      -s dev -s test -s train \
      -d voicemos_main1 \
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
