target_folder=./dataset/PROSTATEx
mkdir ${target_folder}
mkdir ${target_folder}/Image
mkdir ${target_folder}/Mask

source_img_folder=./Raw/PROSTATEx/Image/*/PROSTATEx
source_msk_folder=./Raw/PROSTATEx/Mask/*/PROSTATEx
find $source_msk_folder -mindepth 1 -maxdepth 1 -type d | while read case;
do
  case_id=$(basename ${case})
  echo ${case_id}
  img_path=${source_img_folder}/${case_id}/*/*
  msk_path=${source_msk_folder}/${case_id}/*/*/*.dcm
  python3 utils/dicom_to_nifti.py --dicom_folder ${img_path} --nifti_path ${target_folder}/Image/${case_id}.nii.gz
  mkdir ${target_folder}/Mask/${case_id}/
  utils/segimage2itkimage --inputDICOM ${msk_path} --outputType nifti --outputDirectory ${target_folder}/Mask/${case_id}/
  python3 utils/label_combine.py --ref_image ${target_folder}/Image/${case_id}.nii.gz --input_folder_path ${target_folder}/Mask/${case_id}/ --output_path ${target_folder}/Mask/${case_id}.nii.gz
  rm -r ${target_folder}/Mask/${case_id}/
done