python facecrop.py

# data preprocess in data/vgg_crop

python trainEmbedNet.py --gpu 1 --save_path exps/exp_vgg --train_path data/vgg_crop/train --test_path data/korean_faces/val --test_list data/korean_faces/val_pairs.csv --nClasses 8639 --max_epoch 20 --nOut 1024

# val EER1: 11.29%

python trainEmbedNet.py --gpu 1 --initial_model exps/exp_vgg/epoch0010.model --eval --save_path exps/exp_vgg/test --test_path data/mask_faces/val --test_list data/mask_faces/val_pairs2.csv --nOut 1024

# val EER2: 15.77%

python trainEmbedNet.py --gpu 1 --initial_model exps/exp_vgg/epoch0010.model --save_path exps/exp_vgg_maskfaces_arcface --train_path data/mask_faces/train --test_path data/mask_faces/val --test_list data/mask_faces/val_pairs2.csv --nClasses 200 --max_epoch 100 --trainfunc arcface --nOut 1024 --test_interval 1

# val EER2: 2.72% (18 epoch)

python trainEmbedNet.py --gpu 1 --initial_model exps/exp_vgg_maskfaces_arcface/epoch0018.model --eval --save_path exps/test --test_path data/test_set --test_list data/test_set/test_pairs.csv --output out_vgg_arcface.csv --nOut 1024

# submission EER: 2.898%