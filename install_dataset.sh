source dev/Scripts/activate

curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip --create-dirs ./datasets/kitti/images -o ./datasets/kitti/images/img_kitti.zip
cd ./datasets/kitti/images && unzip -d . img_kitti.zip

curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip --create-dirs ./datasets/kitti/labels -o ./datasets/kitti/labels/lab_kitti.zip
cd ../labels && unzip -d . lab_kitti.zip

# tar 압축 해제
# tar -zxvf [file name] -C [out dir]