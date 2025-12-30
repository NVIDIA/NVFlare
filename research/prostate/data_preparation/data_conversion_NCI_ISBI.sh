source_img_folder=./Raw/NCI_ISBI/Image/
for subset in 3T Dx; do
  target_folder=./dataset/NCI_ISBI_${subset}
  mkdir ${target_folder}
  mkdir ${target_folder}/Image
  mkdir ${target_folder}/Mask
  for split in Training Test Leaderboard; do
    source_msk_folder=./Raw/NCI_ISBI/Mask/${split}
    find $source_msk_folder -mindepth 1 -maxdepth 1 -type f | while read case; do
      msk_id=$(basename ${case})
      msk_id=${msk_id%.*}
      if [[ "${msk_id}" == *"_truth"* ]]; then
        img_id=${msk_id%"_truth"}
      else
        img_id=${msk_id}
      fi
      echo ${img_id} ${msk_id}

      if [[ "${img_id}" == *"${subset}"* ]]; then
        img_path=${source_img_folder}/${img_id}/*/*
        python3 utils/dicom_to_nifti.py --dicom_folder ${img_path} --nifti_path ${target_folder}/Image/${img_id}.nii.gz
        msk_path=${source_msk_folder}/${msk_id}.nrrd
        python3 utils/nrrd_to_nifti.py --input_path ${msk_path} --reference_path ${target_folder}/Image/${img_id}.nii.gz --output_path ${target_folder}/Mask/${img_id}.nii.gz
        python3 utils/label_threshold.py --input_path ${target_folder}/Mask/${img_id}.nii.gz --output_path ${target_folder}/Mask/${img_id}.nii.gz
      fi
    done
  done
done

rm ${target_folder}/Image/ProstateDx-01-0055.nii.gz
rm ${target_folder}/Mask/ProstateDx-01-0055.nii.gz
