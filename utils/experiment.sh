#!/bin/bash
set -eo pipefail
# set -x
# TODO setup slurm cluster -- see https://wiki.ufal.ms.mff.cuni.cz/slurm ask Zdenek who is using it

ROOT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/.." &> /dev/null && pwd )"

hostname="$(hostname)"
printf "Running on hostname $hostname\n"
base_conda_env="/home/oplatek/code/conda/miniconda3_py38_492/etc/profile.d/conda.sh"

printf "\nSourcing base conda environment $base_conda_env \n"
source "$base_conda_env"


conda_env="./env"
printf "Activating conda environment $conda_env\n\n"
conda activate $conda_env

tail_follow_and_notify () {
  logpath="$1"; shift
  job_id="$1"; shift

  touch "$logpath"
  ntf -c ntf-oplatek -t "submitted job_id $job_id. See logpath: tail -f $logpath"
  set +e
  if [ -z "$NO_TAIL_NO_BLOCK" ] ; then
    printf "\tRunning $job_id :\t\t tail -fn500 $logpath\n\n"
    tail -f "$logpath" | sed '/rror:/ q' | sed '/SUCCESSFULLY FINISHED/ q'
    ntf -c ntf-oplatek -t "Job_id $job_id finished with status $?. See $logpath.|$(tail -n3 $logpath)"
  else
    printf "\tRunning $job_id :\n\t\t Run manually tail -fn500 $logpath\n\n"
  fi
}

moose () {
  gpus="$1"; shift
  gpu_ram="$1"; shift
  workers="$1"; shift
  moose_options="$1"; shift

  cmd="moosenet $moose_options -e $experiment_id --gpus $gpus -w $workers"
  printf "\nSubmitting experiment_id $experiment_id with cmd:\n\t$cmd\n"
  # cpus should be +1 from data workers which can be 0 and also wandb should have dedicated cpu
  # cpus=$((workers + 1))  
  # TODO try wandb without additional cpu resource
  cpus=$workers
  node="gpu-*"
  # node="gpu-ms*"
  # node_name="todo"
  # node="gpu.q@gpu-node${node_name}"
  job_name="T${experiment_id}Moose"
    # --gres=gpu:$gpus \
    # --output=$ROOT_DIR/log/%A.txt \
  # --output="$logpath" \
  if [[ "$gpu_ram" == "24G" ]] ; then
    constraint="gpuram24G|gpuram40G|gpuram48G" 
  elif [[ "$gpu_ram" == "40G" ]] ; then
    constraint="gpuram40G|gpuram48G" 
  elif [[ "$gpu_ram" == "48G" ]] ; then
    constraint="gpuram48G"
  else
    printf "\nPlease specify 24G 40G or 48G for gpu ram! Got '$gpu_ram'\n\n"
    exit 2
  fi

  job_id=$(sbatch \
    --partition="gpu-troja,gpu-ms" \
    --dependency="$SLURM_DEPENDENCY_MOOSE" \
    --parsable \
    --no-requeue \
    --mem=36G \
    --job-name=$job_name \
    --cpus-per-gpu $cpus \
    --gpus=$gpus \
    --export=ALL \
    --constraint="$constraint" \
    ./utils/bashrun.sh $cmd \
  )
  if [[ ! -z $SLURM_DEPENDENCY_MOOSE ]] ; then
    printf "\tJob $job_id waits on $SLURM_DEPENDENCY_MOOSE\n"
    unset SLURM_DEPENDENCY_MOOSE
  fi
  logpath="$(realpath $ROOT_DIR/slurm-${job_id}.out)"
  tail_follow_and_notify "$logpath" "$job_id"
}


cpu () {
  cmd="$1"; shift

  cpu_ram=36G
  # cpus="$1"; shift
  printf "\nSubmitting experiment_id $experiment_id with cmd:\n\t$cmd\n"
  # node="gpu-ms*"
  # node_name="todo"
  # node="gpu.q@gpu-node${node_name}"
  job_name="F${job_wait}Moose"

  job_id=$(sbatch \
    --partition="cpu-troja,cpu-ms" \
    --dependency="$SLURM_DEPENDENCY_MOOSE" \
    --parsable \
    --no-requeue \
    --mem=$cpu_ram \
    --job-name=$job_name \
    --export=ALL \
    ./utils/bashrun.sh $cmd \
  )
  if [[ ! -z $SLURM_DEPENDENCY_MOOSE ]] ; then
    printf "\tJob $job_id waits on $SLURM_DEPENDENCY_MOOSE\n"
    unset SLURM_DEPENDENCY_MOOSE
  fi
  logpath="$(realpath $ROOT_DIR/slurm-${job_id}.out)"
  tail_follow_and_notify "$logpath" "$job_id"
}
