import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import time
import torch
import torchvision.transforms.functional as TF
import cv2
from PIL import ImageFile
from mesh_load import *
import pywt

ImageFile.LOAD_TRUNCATED_IMAGES = True


def cv_resize(image, dim):
    image = np.array(image, dtype=np.uint8)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def cv_resize_pil(image, dim):
    image = np.array(image, dtype=np.uint8)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    image = Image.fromarray(image)
    return image


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, mesh_root, lap_root, trainsize, origsize, num_views, fake_back_rate=0,
                 back_dir=None, pb_rate=0, plate_dir=None, with_plate=False, examine_mode=False,
                 back_img=None, test_mode=False, trimap_dir=None, of_list=None, mask_ext='.jpg', max_view=8,
                 with_gray=0, sample_size=None):
        self.sample_size = sample_size
        self.n_view = num_views
        self.lap_root = lap_root
        self.view_list = list(range(max_view))
        self.max_view = max_view
        self.trainsize = trainsize
        self.origsize = origsize
        self.with_plate = with_plate
        self.plate_dir = plate_dir
        self.trimap_dir = trimap_dir
        self.test_mode = test_mode
        self.back_img = back_img
        self.pb_rate = pb_rate
        self.with_gray = with_gray
        self.cameras = camera_setting_loader(os.path.join(image_root, 'default_camera_setting.xml'))

        if num_views > 1:
            self.images = [image_root + f for f in
                           os.listdir(os.path.join(image_root, str(self.cameras['camera_id'][0]))) if
                           f.endswith('.jpg') or f.endswith('.png')]

        else:
            self.images = [image_root + f for f in os.listdir(image_root) if
                           f.endswith('.jpg') or f.endswith('.png')]

        self.mesh_root = mesh_root
        self.mask_ext = mask_ext
        self.image_root = image_root
        self.images = sorted(self.images)
        # self.gts = sorted(self.gts)
        self.size = len(self.images)
        self.totensor = transforms.Compose([
            # transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
        ])
        self.examine_mode = examine_mode
        self.img_transform_after = transforms.Compose([
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        self.fake_back_rate = fake_back_rate
        if trimap_dir:
            with_trimap = True
        else:
            with_trimap = False
        self.fb = FakeBack(back_dir, trainsize=self.trainsize, test_mode=test_mode, back_img=back_img,
                           with_trimap=with_trimap, views=self.n_view)

        if of_list:
            with open(of_list, "r") as f:
                of_list = [i for line in f for i in line.split('\n')]
            of_list = list(filter(None, of_list))
            self.of_list = of_list
        else:
            self.of_list = []

    def check_size(self, image, label):
        force_size = self.origsize
        if not image.size == force_size:
            image = cv_resize_pil(image, force_size)
            label = cv_resize_pil(label, force_size)
        return image, label

    def load_process(self, image_dir, mask_dir):
        image = self.rgb_loader(image_dir)
        if self.test_mode:
            gt = self.binary_loader(image_dir)
        else:
            gt = self.binary_loader(mask_dir)
        image, gt = self.check_size(image, gt)

        # if self.trimap_dir:
        #     trimap = self.binary_loader(self.trimap_dir + filename[:-4] + '.png')

        if self.fake_back_rate + self.pb_rate:
            luck = random.random()
            if image_dir.split('/')[-1] in self.of_list:
                luck = self.fake_back_rate
            if luck > self.fake_back_rate + self.pb_rate:
                orig_image = np.array(image)
                orig_label = np.array(gt)
                image = cv_resize(image, self.trainsize)
                gt = cv_resize(gt, self.trainsize)
                # if self.trimap_dir:
                #     trimap = cv_resize(trimap, (self.trainsize, self.trainsize))
                #     trimap = self.totensor(trimap)
                orig_image = self.totensor(orig_image)
                image = self.totensor(image)
                gt = self.totensor(gt)
                orig_label = self.totensor(orig_label)
            else:
                # if self.trimap_dir:
                #     sample = {'image': image, 'label': gt, 'trimap': trimap}
                # else:
                sample = {'image': image, 'label': gt}
                if luck > self.fake_back_rate and (luck <= self.fake_back_rate + self.pb_rate):
                    sample = self.fb(sample, pb=True)
                else:
                    sample = self.fb(sample, pb=False)
                image = sample['image']
                gt = sample['label']
                orig_image = sample['orig_image']
                orig_label = sample['orig_label']
                if self.with_plate:
                    plate = sample['plate']
                # if self.trimap_dir:
                #     trimap = sample['trimap']

        else:
            orig_image = np.array(image)
            orig_label = np.array(gt)
            image = cv_resize(image, self.trainsize)
            gt = cv_resize(gt, self.trainsize)

            gt = self.totensor(gt)
            orig_image = self.totensor(orig_image)
            image = self.totensor(image)
            orig_label = self.totensor(orig_label)

        if not self.examine_mode:
            image = self.img_transform_after(image)
            orig_image = self.img_transform_after(orig_image)

        if self.with_plate:
            if not self.examine_mode:
                plate = self.img_transform_after(plate)
            image = torch.cat((image, plate), dim=0)

        if random.random() < self.with_gray:
            rgb_weights = [0.2989, 0.5870, 0.1140]
            image_g = rgb_weights[0] * image[0:1, :, :] + rgb_weights[1] * image[1:2, :, :] + rgb_weights[2] * image[
                                                                                                               2:3, :,
                                                                                                               :]
            image = torch.tile(image_g, (3, 1, 1))
            orig_image_g = rgb_weights[0] * orig_image[0:1, :, :] + rgb_weights[1] * orig_image[1:2, :, :] + \
                           rgb_weights[2] * orig_image[2:3, :, :]
            orig_image = torch.tile(orig_image_g, (3, 1, 1))


        return image, gt, orig_image, orig_label

    def p2m_load_img(self, image_dir):
        image = self.rgb_loader(image_dir)
        image = cv_resize(image, self.trainsize)
        image = self.totensor(image)
        if not self.examine_mode:
            image = self.img_transform_after(image)
        return image

    def p2m_load_depth(self, image_dir):
        dmax = 3000
        image = cv2.imread(image_dir, cv2.IMREAD_UNCHANGED)
        image = np.clip(image, 0, dmax)
        image = cv2.resize(image, self.trainsize, interpolation=cv2.INTER_AREA)
        image = np.expand_dims(image, axis=0)
        image = image.astype('float32')
        image = (image/float(dmax)).astype('float32')
        image = torch.from_numpy(image).type(torch.float32)
        return image


    def decomp(self, image, level):
        out_list = []
        coeff = image
        out_list.append(coeff)
        for l in range(level):
            coeff = pywt.dwt2(coeff, 'db1', mode='zero')
            stack = np.concatenate([np.concatenate(coeff[1], axis=1), coeff[0]], axis=1)
            out_list.append(stack)
            coeff = coeff[0]
        return out_list

    def p2m_load_mask(self, image_dir):
        mask = self.binary_loader(image_dir)
        mask = cv_resize(mask, self.trainsize)
        mask = self.totensor(mask)
        return mask


    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')


    def get_multi(self, index):
        step = 1
        images = []
        masks = []
        names = []
        depths = []
        #sample_views = random.sample(self.cameras['camera_id'], k=self.n_view)
        sample_views = self.cameras['camera_id']
        thisfile = self.images[index].split('/')[-1]
        file_index = int(thisfile[5:-4])
        index_views = []
        for v in sample_views:
            index_views.append(self.cameras['camera_id'].index(v))
        this_mesh_name = 'mesh' + str(file_index).zfill(6) + '.ply'
        next_mesh_name = 'mesh' + str(file_index+step).zfill(6) + '.ply'
        filename = 'color'+str(file_index + step).zfill(6)+'.png'

        next_mesh = load_mesh(os.path.join(self.mesh_root, next_mesh_name), self.sample_size)
        if next_mesh['vert'].shape[0] == 0:
            file_index -= step
            this_mesh_name = 'mesh' + str(file_index).zfill(6) + '.ply'
            next_mesh_name = 'mesh' + str(file_index + step).zfill(6) + '.ply'

            next_mesh = load_mesh(os.path.join(self.mesh_root, next_mesh_name), self.sample_size)
            filename = 'color'+str(file_index + step).zfill(6)+'.png'
        next_mesh = load_lap(os.path.join(self.lap_root, 'laplacian'+str(file_index+step).zfill(6)+'.numpy'), next_mesh)
        this_mesh = load_mesh(os.path.join(self.mesh_root, this_mesh_name), self.sample_size)
        this_mesh = load_lap(os.path.join(self.lap_root, 'laplacian'+str(file_index).zfill(6)+'.numpy'), this_mesh)


        for v in sample_views:
            names.append(filename)
            depth_dir = os.path.join(self.image_root, str(v)+'_depth', 'depth'+filename[5:-4]+'.png')
            mask_dir = os.path.join(self.image_root, str(v) + '_mask', filename[:-4] + '.jpg')
            image_dir = os.path.join(self.image_root, str(v), filename[:-4]+'.png')
            image = self.p2m_load_img(image_dir)
            depth = self.p2m_load_depth(depth_dir)
            mask = self.p2m_load_mask(mask_dir)

            #image = image * mask
            #depth = depth * mask
            images.append(image)
            masks.append(mask)
            depths.append(depth)


        image = torch.stack(images, dim=0)
        image = image.float()

        mask = torch.stack(masks, dim=0)
        mask = mask.float()

        depth = torch.stack(depths, dim=0)
        depth = depth.float()

        # img_feats = self.decomp(image, 5)
        # mask_feats = self.decom
        # print(len(decomp_list))
        #image = torch.cat([image, mask], dim=1)
        #image = self.decomp(image, 5)
        return image, this_mesh, next_mesh, names, index_views, mask, depth

    # def get_single(self, index):
    #
    #     filename = self.images[index].split('/')[-1]
    #     image_dir = os.path.join(self.image_root, filename)
    #     mask_dir = os.path.join(self.gt_root, filename[:-4]) + self.mask_ext
    #
    #     image, gt, orig_image, orig_label = self.load_process(image_dir, mask_dir)
    #
    #     image = image.float()
    #     gt = gt.float()
    #     orig_image = orig_image.float()
    #     orig_label = orig_label.float()
    #
    #     return image, gt, orig_image, orig_label, filename

    def __getitem__(self, index):
        num = 4
        max_ind = len(self.images) - num - 1
        sequence = []
        if index > max_ind:
            index = random.randint(0, max_ind)

        for i in range(num):
            image, this_mesh, next_mesh, names, sample_views, mask, depth = self.get_multi(index + i)
            sequence.append({'image': image, 'this_mesh': this_mesh, 'next_mesh':  next_mesh, 'names': names, 'sample_views': sample_views, 'mask': mask, 'depth': depth})

        # if self.test_mode:
        #     return image, this_mesh, next_mesh, names, sample_views, mask, depth
        # else:
        #     return image, this_mesh, next_mesh, names, sample_views, mask, depth
        return sequence

    def filter_files(self):
        # print(len(self.images), len(self.gts))
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images = images
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')



    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(data_root, mesh_root, lap_root, batchsize, trainsize, origsize, n_views, shuffle=True, num_workers=24,
               pin_memory=True, sample_size=None,
               fake_back_rate=0, back_dir=None, back_img=None, pure_back_rate=0, with_plate=False,
               examine_mode=False, trimap_dir=None, of_list=None, with_gray=0, mask_ext='.jpg'):
    if back_img:
        test_mode = True
    else:
        test_mode = False

    dataset = SalObjDataset(data_root, mesh_root, trainsize=trainsize, origsize=origsize, num_views=n_views,
                            fake_back_rate=fake_back_rate,
                            back_dir=back_dir, pb_rate=pure_back_rate, plate_dir=None, with_plate=with_plate,
                            examine_mode=examine_mode, back_img=back_img, test_mode=test_mode, trimap_dir=trimap_dir,
                            of_list=of_list, mask_ext=mask_ext, with_gray=with_gray, sample_size=sample_size, lap_root=lap_root)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory,
                                  worker_init_fn=worker_init_fn
                                  )
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, plate_root, testsize, orig=False, w_plate=False):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root)]
        plate = self.rgb_loader(plate_root)
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.w_plate = w_plate
        self.index = 0
        self.orig = orig
        self.plate = self.transform(plate).unsqueeze(0)

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])

        if self.orig:
            image_orig = image.copy()
        image = self.transform(image).unsqueeze(0)
        print(image.shape)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]

        self.index += 1
        if self.w_plate:
            image = torch.cat((image, self.plate), dim=1)
        if self.orig:
            return np.array(image_orig), image, gt, name
        else:
            return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


class FakeBack(object):

    def __init__(self, back_dir, trainsize, views, back_img=None, test_mode=False, with_trimap=False):
        import torchvision.transforms as T
        self.back_dir = back_dir
        self.test_mode = test_mode
        self.back_img = back_img
        self.with_trimap = with_trimap
        self.n_view = views
        self.trainsize = trainsize
        if self.back_dir:
            path, dirs, files = next(os.walk(back_dir))
            random.shuffle(files)
            num = len(files)
            self.selected_backs = files[:int(num * 1.)]

        path, dirs, files = next(os.walk('/home/hypevr/Desktop/datasets/background/studio/'))
        random.shuffle(files)
        self.studio = files
        self.autocontrast = T.RandomAutocontrast(p=0.5)
        self.sharpness_adjuster = T.RandomAdjustSharpness(sharpness_factor=2, p=0.5)
        self.equalizer = T.RandomEqualize(p=0.5)
        self.blurrer = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
        self.perspective_transformer = T.RandomPerspective(distortion_scale=0.5, p=0.5)
        # self.randomcrop = T.RandomCrop(, padding=None, pad_if_needed=False, fill=0, padding_mode='constant')

    def cv_resize(self, image):
        image = cv2.resize(image, self.trainsize, interpolation=cv2.INTER_AREA)
        return image

    def random_resize_crop(self, image, crop_pct, out_mode='same', out_size=None, p=0.8):
        if crop_pct > 1:
            raise Exception('crop percentage should be less than 1.0')

        w, h = image.size
        new_w = int(w * crop_pct)
        new_h = int(h * crop_pct)

        if out_mode == 'same':
            out_size = (w, h)
        elif out_mode == 'fixed':
            if not out_size:
                raise Exception('if fixed mode is chosen, out_size has to be specified')
            else:
                out_size = out_size
        elif out_mode == 'no_resize':
            out_size = (new_w, new_h)
        else:
            raise Exception('out_mode error: choose from "fixed", "same" and "no_resize"')

        prob = random.random()
        if prob > p:
            out_image = cv_resize_pil(image, out_size)
        else:
            pos_h = random.randint(0, h - new_h)
            pos_w = random.randint(0, w - new_w)
            out_image = TF.crop(image, top=pos_h, left=pos_w, height=new_h, width=new_w)
            out_image = cv_resize_pil(out_image, out_size)
        return out_image

    def totensor(self, image):
        image = image / 255.
        if len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        elif len(image.shape) == 2:
            image = np.expand_dims(image, axis=0)
        image = torch.tensor(image)
        return image

    def back_transform(self, back, dim):

        # angle = random.randint(-10, 10)
        # shear = random.uniform(-10, 10)

        # back = TF.affine(back, angle=0, translate=[0, 0], fill=0, shear=shear, scale=1)

        if random.random() > 0.5:
            back = TF.hflip(back)

        if random.random() > 0.5:
            back = TF.vflip(back)

        if random.random() > 0.2:
            back = self.autocontrast(back)
            back = self.sharpness_adjuster(back)
            back = self.equalizer(back)

        rotate_p = random.random()
        if rotate_p < 0.1:
            back = back.rotate(90)
        elif 0.1 <= rotate_p < 0.2:
            back = back.rotate(270)

        back = self.random_resize_crop(back, crop_pct=random.uniform(0.4, 0.9), out_mode='fixed', out_size=dim, p=0.8)

        return back

    def __call__(self, sample, pb):

        image, label = sample['image'], sample['label']
        orig_w, orig_h = image.size
        if self.with_trimap:
            trimap = sample['trimap']

        else:
            trimap = label

        if not self.test_mode:
            random.seed(time.time())
            trans_r_v = 0.05
            trans_r_h = 0.4
            rand_shift_v = int(orig_h * trans_r_v)
            rand_shift_h = int(orig_w * trans_r_h)

            zoom = random.uniform(0.7, 1.2)
            angle = random.randint(-25, 25)
            translate = [random.randint(-rand_shift_h, rand_shift_h),
                         random.randint(-rand_shift_v, rand_shift_v)]

            zoom_size_w = int(orig_w * zoom)
            if zoom_size_w % 2 == 1:
                zoom_size_w += 1
            zoom_size_h = int(orig_h * zoom)
            if zoom_size_h % 2 == 1:
                zoom_size_h += 1
            zoom_size = (zoom_size_w, zoom_size_h)

            image = cv_resize_pil(image, zoom_size)
            label = cv_resize_pil(label, zoom_size)
            trimap = cv_resize_pil(trimap, zoom_size)

            if zoom > 1:
                image = TF.center_crop(image, [orig_h, orig_w])
                label = TF.center_crop(label, [orig_h, orig_w])
                trimap = TF.center_crop(trimap, [orig_h, orig_w])
            else:
                pad_w = int((orig_w - zoom_size[0]) / 2)
                pad_h = int((orig_h - zoom_size[1]) / 2)
                image = TF.pad(image, padding=[pad_w, pad_h, pad_w, pad_h])
                label = TF.pad(label, padding=[pad_w, pad_h, pad_w, pad_h])
                trimap = TF.pad(trimap, padding=[pad_w, pad_h, pad_w, pad_h])

            image = TF.affine(image, angle=angle, translate=translate, fill=0, shear=[0, 0], scale=1)
            label = TF.affine(label, angle=angle, translate=translate, fill=0, shear=[0, 0], scale=1)
            trimap = TF.affine(trimap, angle=angle, translate=translate, fill=0, shear=[0, 0], scale=1)

            image = self.equalizer(image)
            image = self.autocontrast(image)
            image = self.sharpness_adjuster(image)

            if random.random() > 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                trimap = TF.hflip(trimap)

            if random.random() > 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                trimap = TF.vflip(trimap)

            if random.random() > 0.8:
                back = Image.open(self.back_dir + random.choice(self.selected_backs))
                back = back.convert('RGB')
                back = self.back_transform(back, (720, 1280))
            else:
                # back = Image.open(self.back_dir + random.choice(self.selected_backs))
                back = Image.open('/home/hypevr/Desktop/datasets/background/studio/' + random.choice(self.studio))
                back = back.convert('RGB')
                back = self.back_transform(back, (720, 1280))

        else:
            back = Image.open(self.back_img)
            back = back.convert('RGB')
        back = np.array(back)

        image = np.array(image)
        label = np.array(label)
        trimap = np.array(trimap)
        if np.sum(label / 255.) / (orig_w * orig_h) < 0.01:
            pb = True
        # orig_trimap = np.array(trimap)
        label_cp = label.copy()

        if not self.test_mode:
            label_cp = 255. * (label_cp.astype(np.float32) / float(label_cp.max() + 1e-8))
            if not self.with_trimap:
                label_cp[label_cp < 100] = 0
                # label_cp[np.logical_and(50 <= label, label < 128)] = 128
                label_cp[label_cp >= 100] = 255

            mask = np.tile(np.expand_dims(label_cp, axis=-1), (1, 1, 3)) / 255.
            image = image * mask  # ).astype('uint8')
            # kernel = np.ones((3, 3), np.uint8)#cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # image = cv2.erode(image, kernel, iterations=2)
            # image = image * mask
            olay = image.copy()
            compare = np.all(image == (0, 0, 0), axis=-1)

            ##randomly dim the image to match the brightness of the background
            random_dim = random.uniform(0.7, 1)
            olay = olay * random_dim

            ##random perspective transform
            # olay = torch.tensor(olay)
            # olay = self.perspective_transformer(olay)
            # olay = np.array(olay)

            ##Apply random gaussian noise to the image
            noise = np.random.normal(0, 1, (orig_h, orig_w, 3))

            ##apply the overlay
            olay[compare] = back[compare]

            olay = cv2.bilateralFilter(olay, 3, 75, 75)

            olay = np.clip(olay + noise, 0, 255)

        else:
            olay = image  # / 255.

        plate = back
        back = np.clip(back + noise, 0, 255)
        orig_olay = olay
        orig_label = label_cp
        orig_back = back

        orig_olay = self.totensor(orig_olay)
        orig_label = self.totensor(orig_label)
        orig_back = self.totensor(orig_back)

        olay = self.cv_resize(olay)
        olay = self.totensor(olay)
        back = self.cv_resize(back)
        back = self.totensor(back)
        trimap = self.cv_resize(trimap)
        trimap = self.totensor(trimap)
        label_cp = self.cv_resize(label_cp)
        label_cp = self.totensor(label_cp)

        trimap[trimap < 64] = 0
        trimap[np.logical_and(64 <= trimap, trimap < 192)] = 1
        trimap[trimap >= 192] = 2

        trimap = np.expand_dims(trimap, axis=0)
        # trimap = torch.tensor(trimap)

        plate = self.totensor(plate)

        if not pb:
            return {'image': olay, 'label': label_cp, 'plate': plate, 'trimap': trimap, 'orig_image': orig_olay,
                    'orig_label': orig_label}
        else:
            return {'image': back, 'label': torch.zeros((1, self.trainsize[0], self.trainsize[1])),
                    'plate': plate, 'trimap': torch.zeros((1, self.trainsize[0], self.trainsize[1])),
                    'orig_image': orig_back, 'orig_label': torch.zeros((1, orig_h, orig_w))
                    }


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2 ** 30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2 ** 30
    np.random.seed(torch_seed + worker_id)
