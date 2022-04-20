source dev/Scripts/activate

curl https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip -create-dir ./datasets/kitti -o ./datasets/kitti/kitti.zip
unzip -d . kitti.zip

# tar 압축 해제
# tar -zxvf [file name] -C [out dir]