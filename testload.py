import mesh_load
from multi_data import get_loader
from torch_geometric import utils
import torch

train_data_loader = get_loader(data_root='/home/hypevr/nas/2021/08/sequence_01/Video_09082021_145444/',
                                           mesh_root='/home/hypevr/nas/2021/09/0927/deform_data_set_02/mesh/',
                                           batchsize=1,
                                           trainsize=224,
                                           origsize=None,
                                           n_views=4,
                                           shuffle=True,
                                           num_workers=20,
                                           pin_memory=False,
                                           fake_back_rate=0,
                                           back_dir=None,
                                           back_img=None,
                                           pure_back_rate=0,
                                           with_plate=False,
                                           examine_mode=False,
                                           trimap_dir=None,
                                           of_list=None,
                                           with_gray=0,
                                           mask_ext='.jpg', sample_size=None)

for step, batch in enumerate(train_data_loader):
    image, this_mesh, next_mesh, sample_views = batch
    #print(this_mesh['edges'].shape)
    #matrix = torch.from_numpy(this_mesh['edges'])
    print(this_mesh['vert'].shape)
    matrix = utils.to_dense_adj(torch.squeeze(this_mesh['edge']))
    #print(image.shape)
    input('wait')
# cameras = mesh_load.camera_setting_loader('/home/hypevr/nas/2021/08/sequence_01/Video_09082021_145444/default_camera_setting.xml')
# print(cameras)