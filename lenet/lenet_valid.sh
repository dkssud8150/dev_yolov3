source dev/Scripts/activate

python lenet5.py --name fashionmnist --mode eval --download 0 --checkpoint './output/fashionmnist/models/best_epoch.pt' --device gpu