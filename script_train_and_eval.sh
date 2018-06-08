#!/bin/bash

############################## use simple huge kernel 21 and huge mode 1 #################################
#21 no image global pooling, with dropout, depth=1024
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt \
    --train_logdir=./deeplab/train_log/21_train_dir \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.007 \
    --fine_tune_batch_norm=True \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=False \
    --new_module_depth=1024

# 22 
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg


python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=15000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/21_train_dir/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/22_train_dir \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=False \
    --new_module_depth=1024

#23 no image global pooling, with dropout, depth=512
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt \
    --train_logdir=./deeplab/train_log/23_train_dir \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.007 \
    --fine_tune_batch_norm=True \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=False \
    --new_module_depth=512

# 24
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg


python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=15000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/23_train_dir/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/24_train_dir \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=False \
    --new_module_depth=512



<<COMMENT
############################## use simple huge kernel 21 and huge mode 1 #################################
#17 no image global pooling
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt \
    --train_logdir=./deeplab/train_log/17_train_dir_true_no_decoder_no_aspp_huge_21_huge_mode_1_no_ipooling_aug_0.007 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.007 \
    --fine_tune_batch_norm=True \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=False

# 18 
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg


python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=10000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/17_train_dir_true_no_decoder_no_aspp_huge_21_huge_mode_1_no_ipooling_aug_0.007/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/18_train_dir_false_no_decoder_no_aspp_huge_21_huge_mode_1_no_ipooling_voc_0.001 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=False

#19 with image global pooling
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt \
    --train_logdir=./deeplab/train_log/19_train_dir_true_no_decoder_no_aspp_huge_21_huge_mode_1_with_ipooling_aug_0.007 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.007 \
    --fine_tune_batch_norm=True \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=True

# 20 
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg


python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=10000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/19_train_dir_true_no_decoder_no_aspp_huge_21_huge_mode_1_with_ipooling_aug_0.007/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/20_train_dir_false_no_decoder_no_aspp_huge_21_huge_mode_1_with_ipooling_voc_0.001 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False \
    --huge_kernel_size=21 \
    --huge_mode=1 \
    --add_image_level_feature=True
COMMENT
############################## use huge kernel 31 and new module depth 256 #################################
#13
#mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
#mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg
<<COMMENT
python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt \
    --train_logdir=./deeplab/train_log/13_train_dir_true_no_decoder_no_aspp_huge_21_depth_256_aug_0.007 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.007 \
    --fine_tune_batch_norm=True \
    --huge_kernel_size=21
    --new_module_depth=256

# 14
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg
COMMENT

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/13_train_dir_true_no_decoder_no_aspp_huge_21_depth_256_aug_0.007/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/14_train_dir_false_no_decoder_no_aspp_huge_21_depth_256_voc_0.001 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False \
    --huge_kernel_size=21 \
    --new_module_depth=256
<<COMMENT
############################## use huge kernel 31 and new module depth 512 #################################
#15
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt \
    --train_logdir=./deeplab/train_log/15_train_dir_true_no_decoder_no_aspp_huge_31_depth_512_aug_0.007 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.007 \
    --fine_tune_batch_norm=True \
    --huge_kernel_size=31
    --new_module_depth=512

# 16
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/15_train_dir_true_no_decoder_no_aspp_huge_31_depth_512_aug_0.007/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/16_train_dir_false_no_decoder_no_aspp_huge_31_depth_512_voc_0.001 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False \
    --huge_kernel_size=31 \
    --new_module_depth=512
COMMENT
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --dataset="pascal_voc_seg" \
    --checkpoint_dir=./deeplab/train_log/13_train_dir_true_no_decoder_no_aspp_huge_21_depth_256_aug_0.007 \
    --eval_logdir=./deeplab/eval_log \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --huge_kernel_size=21 \
    --new_module_depth=256

python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --dataset="pascal_voc_seg" \
    --checkpoint_dir=./deeplab/train_log/14_train_dir_false_no_decoder_no_aspp_huge_21_depth_256_voc_0.001 \
    --eval_logdir=./deeplab/eval_log \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --huge_kernel_size=21 \
    --new_module_depth=256
<<COMMENT
python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --dataset="pascal_voc_seg" \
    --checkpoint_dir=./deeplab/train_log/15_train_dir_true_no_decoder_no_aspp_huge_31_depth_512_aug_0.007 \
    --eval_logdir=./deeplab/eval_log \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --huge_kernel_size=31 \
    --new_module_depth=512

python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --eval_crop_size=513 \
    --eval_crop_size=513 \
    --dataset="pascal_voc_seg" \
    --checkpoint_dir=./deeplab/train_log/16_train_dir_false_no_decoder_no_aspp_huge_31_depth_512_voc_0.001 \
    --eval_logdir=./deeplab/eval_log \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --huge_kernel_size=31 \
    --new_module_depth=512
COMMENT
<<COMMENT
############################## 20180606 ################################
############################## performance in different training steps with original network #################################
# 11     
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg


python deeplab/train.py     --logtostderr     --training_number_of_steps=30000     --train_split="train"         --model_variant="xception_65"     --atrous_rates=6     --atrous_rates=12     --atrous_rates=18         --output_stride=16  --decoder_output_stride=4    --train_crop_size=513     --train_crop_size=513         --train_batch_size=16     --dataset="pascal_voc_seg"     --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt     --train_logdir=./deeplab/train_log/train_dir_true_aug_0.007_more_save    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord --fine_tune_batch_norm=True --base_learning_rate=0.007

# 12
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/train_dir_true_aug_0.007_more_save/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/train_dir_false_voc_0.001_more_save \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False


############################## performance without decoder #################################
# 7
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg


python deeplab/train.py     --logtostderr     --training_number_of_steps=30000     --train_split="train"         --model_variant="xception_65"     --atrous_rates=6     --atrous_rates=12     --atrous_rates=18         --output_stride=16      --train_crop_size=513     --train_crop_size=513         --train_batch_size=16     --dataset="pascal_voc_seg"     --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt     --train_logdir=./deeplab/train_log/train_dir_true_no_decoder_aug_0.007    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord --fine_tune_batch_norm=True --base_learning_rate=0.007

# 8
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/train_dir_true_no_decoder_aug_0.007/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/train_dir_false_no_decoder_voc_0.001 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False


############################## performance without decoder and aspp #################################
# 9
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_real
mv ./deeplab/datasets/pascal_voc_seg_aug ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/init_checkpoint/xception/model.ckpt \
    --train_logdir=./deeplab/train_log/train_dir_true_no_decoder_no_aspp_aug_0.007 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.007 \
    --fine_tune_batch_norm=True

# 10
mv ./deeplab/datasets/pascal_voc_seg ./deeplab/datasets/pascal_voc_seg_aug
mv ./deeplab/datasets/pascal_voc_seg_real ./deeplab/datasets/pascal_voc_seg

python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --output_stride=16 \
    --train_crop_size=513 \
    --train_crop_size=513 \
    --train_batch_size=16 \
    --num_clones=4 \
    --dataset="pascal_voc_seg" \
    --tf_initial_checkpoint=./deeplab/train_log/train_dir_true_no_decoder_no_aspp_aug_0.007/model.ckpt-30000 \
    --train_logdir=./deeplab/train_log/train_dir_false_no_decoder_no_aspp_voc_0.001 \
    --dataset_dir=./deeplab/datasets/pascal_voc_seg/tfrecord \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=False



COMMENT
