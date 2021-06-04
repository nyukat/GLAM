#!/bin/bash

# Required parameters (GPU_NUMBER only required if DEVICE_TYPE=gpu)
INPUT_FOLDER=${1:-sample_data} # Folder with all input files
OUTPUT_FOLDER=${2:-sample_output} # Folder, where visualization files and predictions will be saved
MODEL_NAME=${3:-model_sep} # Name of the model to use. Valid values include {`model_joint`, `model_sep`}
DEVICE_TYPE=${4:-cpu}  # Either `gpu` or `cpu`.
GPU_NUMBER=${5:-0}  # Which gpu to use, e.g. `0`, does not matter if `DEVICE_TYPE==cpu`.

# Parameters that, most likely, will not need to be changed
MODEL_PATH="models/"
IMAGES_FOLDER="${INPUT_FOLDER}/images"
INITIAL_EXAM_LIST_PATH="${INPUT_FOLDER}/exam_list_before_cropping.pkl"
CROPPED_IMAGE_FOLDER="${INPUT_FOLDER}/cropped_images"
CROPPED_EXAM_LIST_PATH="${INPUT_FOLDER}/cropped_images/cropped_exam_list.pkl"
SEG_FOLDER="${INPUT_FOLDER}/segmentation"
FINAL_EXAM_LIST_PATH="${INPUT_FOLDER}/data.pkl"
NUM_PROCESSES=10
RUN_TIME="$(date +"%Y-%m-%d-%T")"
OUTPUT_FINAL_FOLDER="${OUTPUT_FOLDER}/${RUN_TIME}"

mkdir -p "${OUTPUT_FINAL_FOLDER}" > /dev/null
export PYTHONPATH=$(pwd):$PYTHONPATH

echo -e "\nStage 1: Cropping Mammograms.."
python3 src/cropping/crop_mammogram.py \
    --input-data-folder "${IMAGES_FOLDER}" \
    --output-data-folder "${CROPPED_IMAGE_FOLDER}" \
    --exam-list-path "${INITIAL_EXAM_LIST_PATH}"  \
    --cropped-exam-list-path "${CROPPED_EXAM_LIST_PATH}"  \
    --num-processes "${NUM_PROCESSES}"

echo -e "\nStage 2: Extracting Centers.."
python3 src/optimal_centers/get_optimal_centers.py \
    --cropped-exam-list-path "${CROPPED_EXAM_LIST_PATH}" \
    --data-prefix "${CROPPED_IMAGE_FOLDER}" \
    --output-exam-list-path "${FINAL_EXAM_LIST_PATH}" \
    --num-processes "${NUM_PROCESSES}"

echo -e "\nStage 3: Running Classifier.."
python3 src/scripts/run_model.py \
    --model-path "${MODEL_PATH}" \
    --data-path "${FINAL_EXAM_LIST_PATH}" \
    --image-path "${CROPPED_IMAGE_FOLDER}" \
    --segmentation-path "${SEG_FOLDER}" \
    --output-path "${OUTPUT_FINAL_FOLDER}" \
    --device-type "${DEVICE_TYPE}" \
    --gpu-number "${GPU_NUMBER}" \
    --model-index "${MODEL_NAME}" \
    --visualization-flag
