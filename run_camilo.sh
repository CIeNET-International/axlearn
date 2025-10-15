#!/usr/bin/env bash
set -xe

export RANDOM_CHARS=$(LC_CTYPE=C openssl rand -base64 12 | tr -dc 'a-z0-9' | head -c 3 ; echo)
#export PROJECT_ID="cloud-tpu-best-effort-colo"
export PROJECT_ID="cloud-tpu-multipod-dev"
export NUM_REPLICAS=${NUM_REPLICAS:-2}
export JOBSET_NAME=axlearn-image-test-$RANDOM_CHARS
#export JOBSET_NAME="sujinesh-64-orbax-3"
export BASTION_TIER=disabled
export GKE_CLUSTER=$(axlearn gcp config | grep gke_cluster | awk '{ print $3 }' | tr -d '"')
# Switch to tpu-v6e-256 if on scale cluster
export INSTANCE_TYPE=${INSTANCE_TYPE:-"tpu-v5p-128"}
# Switch to tpu-v6e-256-4 if on scale cluster
export MESH_SELECTOR=${MESH:-"tpu-v5p-128"}
export CONFIG=${CONFIG:-"fuji-7B-v2-flash"}
#export OUTPUT_DIR=${OUTPUT_DIR:-gs://tess-checkpoints-flat-us-east5}
export OUTPUT_DIR="gs://axlearn-ml-solutions"
# export DATA_DIR="gs://tess-apple-southamerica-west1/tensorflow_datasets"
#export DATA_DIR="gs://tess-dataloading-us-east5/tensorflow_datasets"
export DATA_DIR="gs://axlearn-public/tensorflow_datasets"


# Example for v6e-256
# MESH_SELECTOR=tpu-v6e-256-4 INSTANCE_TYPE=tpu-v6e-256 ./test-orbax.sh

# The bundle step is needed if you run on cloudtop
# uncomment if you use cloudtop
axlearn gcp bundle --name=$JOBSET_NAME \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry \
        --bundler_spec=dockerfile=Dockerfile \
        --bundler_spec=image=tpu \
        --bundler_spec=target=tpu

# Only enable kueue when running on scale testing cluster
# --queue=multislice-queue \
# --priority_class=very-high \
# --trainer_dir=gs://tess-checkpoints-us-west1/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
#

# Check if CONFIG ends with "orbaxem"
if [[ "$CONFIG" == *"orbaxem"* ]]; then
  echo "Running with Orbax emergency checkpointer."
  axlearn gcp launch run --cluster=$GKE_CLUSTER \
        --runner_name gke_tpu_single \
        --name=$JOBSET_NAME \
        --instance_type=${INSTANCE_TYPE} \
        --host_mount_spec=name=tmp,host_path=/tmp,mount_path=/host-tmp \
        --num_replicas=${NUM_REPLICAS} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "patch /opt/venv/lib/python3.10/site-packages/jax/experimental/shard_map.py -p0 < patches/shard_map.py.patch; ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
          --init_module=axlearn.common.checkpointer_orbax_emergency:local_ckpt_dir=/host-tmp/checkpoints \
          --module=text.gpt.c4_trainer \
          --config=${CONFIG} \
          --trainer_dir=gs://${PROJECT_ID}-axlearn/${JOBSET_NAME}-nr-${NUM_REPLICAS}/ \
          --data_dir=gs://axlearn-public/tensorflow_datasets  \
          --jax_backend=tpu \
          --mesh_selector=${MESH_SELECTOR} \
          --initialization_timeout=1200 \
          --trace_at_steps=29,59,89,119,149,179,209,239,269,299,329,359,389,419,449,479,509,539,569,599,629,659,689,719

else
  echo "Running Orbax regular checkpointer or AXLearn native."
  axlearn gcp launch run --cluster=$GKE_CLUSTER \
        --runner_name gke_tpu_single \
        --name=$JOBSET_NAME \
        --max_tries=100 \
        --instance_type=${INSTANCE_TYPE} \
        --num_replicas=${NUM_REPLICAS} \
        --bundler_spec=allow_dirty=True \
        --bundler_type=artifactregistry --bundler_spec=image=tpu \
        --bundler_spec=dockerfile=Dockerfile --bundler_spec=target=tpu \
        -- "ulimit -n 1048576; ulimit -c 0; python3 -c 'import jax; jax.devices()'; python3 -m axlearn.common.launch_trainer_main" \
          --module=text.gpt.c4_trainer \
          --config=${CONFIG} \
          --trainer_dir=${OUTPUT_DIR}/${JOBSET_NAME}/ \
          --data_dir=$DATA_DIR \
          --jax_backend=tpu \
          --mesh_selector=${MESH_SELECTOR} \
          --initialization_timeout=1200 \
          --trace_at_steps=50,101
fi


        # "ulimit -n 1048576; ulimit -c 0; sed -i 's|max_step = TOTAL_TOKENS\\[version\\]\\[model_size\\] // tokens_per_batch|max_step = 200|; /max_step = 200/a \\    save_every_n_steps=50' axlearn/experiments/text/gpt/fuji.py;python3 -c 'import jax; jax.devices()';
