target_folder=./dataset/MSD
mkdir ${target_folder}
mkdir ${target_folder}/Image
mkdir ${target_folder}/Mask

source_folder=./Raw/MSD/Task05_Prostate
find ${source_folder}/imagesTr -mindepth 1 -maxdepth 1 -type f | while read case; do
  case=$(basename "${case}")
  case="${case%.*.*}"
  echo ${case}

  img_path=${source_folder}/imagesTr/${case}.nii.gz
  python3 utils/image_channel_select.py --input_path ${img_path} --output_path ${target_folder}/Image/${case}.nii.gz
  msk_path=${source_folder}/labelsTr/${case}.nii.gz
  python3 utils/label_threshold.py --input_path ${msk_path} --output_path ${target_folder}/Mask/${case}.nii.gz
done