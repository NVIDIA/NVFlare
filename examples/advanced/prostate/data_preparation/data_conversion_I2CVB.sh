target_folder=./dataset/I2CVB
mkdir ${target_folder}
mkdir ${target_folder}/Image
mkdir ${target_folder}/Mask

for subset in GE Siemens; do
  source_folder=./Raw/I2CVB/${subset}
  find ${source_folder} -mindepth 1 -maxdepth 1 -type d | while read case; do
    case=$(basename "${case}")
    echo ${case}
    case_no_space=$(echo "${case}" | tr ' ' '_')
    mv ${source_folder}/"${case}" ${source_folder}/"${case_no_space}"

    img_path=${source_folder}/${case_no_space}/T2W
    msk_path=${source_folder}/${case_no_space}/GT/prostate

    python3 utils/dicom_to_nifti.py --dicom_folder ${img_path} --nifti_path ${target_folder}/Image/${case_no_space}.nii.gz
    python3 utils/dicom_to_nifti.py --dicom_folder ${msk_path} --nifti_path ${target_folder}/Mask/${case_no_space}.nii.gz
    python3 utils/label_threshold.py --input_path ${target_folder}/Mask/${case_no_space}.nii.gz --output_path ${target_folder}/Mask/${case_no_space}.nii.gz
  done
done

rm ${target_folder}/Image/Patient_428260.nii.gz
