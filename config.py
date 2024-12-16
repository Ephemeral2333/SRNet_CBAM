mode = 'train'  # train or test
epochs = 50
# optimizer: Adam
lr = 2e-4
weight_decay = 1e-5
gamma = 0.5
weight_decay_step = 15

train_batch_size = 16
val_batch_size = 16
test_batch_size = 4
save_freq = 2
val_freq = 2
start_save_epoch = 2

train_data_dir = '/root/autodl-tmp/HUGO4_dataset/train/'
val_data_dir = '/root/autodl-tmp/HUGO4_dataset/validation/'
# train_data_dir = 'D:\\File\\Graduation Design\\dataset\\wow7_dataset\\train'
# val_data_dir = 'D:\\File\\Graduation Design\\dataset\\wow7_dataset\\validation'
test_data_dir = '/root/autodl-tmp/wow1_dataset/test/'
stego_img_height = 256  # stego_img_height == stego_img_width 长宽
stego_img_channel = 3  # 彩色3 灰色1

pre_trained_srnet_path = '/root/autodl-tmp/srnet_02/checkpoints/SRNet/checkpoint_hugo4.pt'
# pre_trained_srnet_path = None