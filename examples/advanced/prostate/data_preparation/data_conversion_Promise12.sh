target_folder=./dataset/Promise12
mkdir ${target_folder}
mkdir ${target_folder}/Image
mkdir ${target_folder}/Mask

source_folder=./Raw/Promise12/Raw_Data
for case_id in {00..49};
do
  img_id="Case${case_id}"
  msk_id="Case${case_id}_segmentation"
  echo ${img_id} ${msk_id}
  img_path=${source_folder}/${img_id}.mhd
  msk_path=${source_folder}/${msk_id}.mhd
  python3 utils/mhd_to_nifti.py --input_path ${img_path} --output_path ${target_folder}/Image/${img_id}.nii.gz
  python3 utils/mhd_to_nifti.py --input_path ${msk_path} --output_path ${target_folder}/Mask/${img_id}.nii.gz
done