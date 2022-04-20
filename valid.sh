source dev/Scripts/activate

python img_classify.py --name fashionmnist --mode eval --download 0 --checkpoint './output/fashionmnist/models/best_epoch.pt' --device gpu