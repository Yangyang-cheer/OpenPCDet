import copy
import pickle
import os
import tqdm
import open3d as o3d
import numpy as np
import open3d
import yaml
from pathlib import Path
from easydict import EasyDict
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
import json
from PIL import Image

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.camera_config = self.dataset_cfg.get('CAMERA_CONFIG', None)
        if self.camera_config is not None:
            self.use_camera = self.camera_config.get('USE_CAMERA', True)
            self.camera_image_config = self.camera_config.IMAGE
        else:
            self.use_camera = False

        self.split_dir = os.path.join(self.root_path, 'ImageSets', (self.split + '.txt'))
        self.sample_id_list = []
        self.custom_infos = []
        self.include_data(self.mode)
        self.map_class_to_kitti = self.dataset_cfg.MAP_CLASS_TO_KITTI
        self.cam_type= ["left","right","bleft","bright"]
        self.split_file=[]
        self.calib_data
    def load_calib_data(self):
        for cam_type in self.cam_type:
            cam_calib_data={}
            
    def include_data(self, mode):
        self.logger.info('Loading Custom dataset.')
        file_path=str(self.root_path)+"/calib_data.json"
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)
        self.custom_infos.extend(custom_infos)
        self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))
        with open(file_path, 'r') as f:
            self.calib_data= json.load(f)


    def get_label(self,idx,bagpath):
        label_file = Path(bagpath + '/labels/'+ idx +'.txt')
        
        assert label_file.exists()
        with open(label_file, 'r') as f:
            lines = f.readlines()

        # [N, 8]: (x y z dx dy dz heading_angle category_id)
        gt_boxes = []
        gt_names = []
        for line in lines:
            line_list = line.strip().split(' ')
            gt_boxes.append(line_list[:-1])
            gt_names.append(line_list[-1])

        return np.array(gt_boxes, dtype=np.float32), np.array(gt_names)
    def crop_image(self, input_dict):
        W, H = input_dict["ori_shape"]
        imgs = input_dict["camera_imgs"]
        img_process_infos = []
        crop_images = []
        for img in imgs:
            if self.training == True:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TRAIN
                resize = np.random.uniform(*resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(np.random.uniform(0, max(0, newW - fW)))
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            else:
                fH, fW = self.camera_image_config.FINAL_DIM
                resize_lim = self.camera_image_config.RESIZE_LIM_TEST
                resize = np.mean(resize_lim)
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = newH - fH
                crop_w = int(max(0, newW - fW) / 2)
                crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            
            # reisze and crop image
            img = img.resize(resize_dims)
            img = img.crop(crop)
            crop_images.append(img)
            img_process_infos.append([resize, crop, False, 0])
        
        input_dict['img_process_infos'] = img_process_infos
        input_dict['camera_imgs'] = crop_images
        return input_dict
    
    def get_lidar(self, lidar_path):
   
        assert Path(lidar_path).exists()
        pcd = o3d.t.io.read_point_cloud(lidar_path)
        
        xyz = pcd.point["positions"].numpy()
        intensity = pcd.point["intensity"].numpy()
        point_features = np.concatenate([xyz, intensity], axis=1)
      
        return point_features
    
    def load_camera_info(self, input_dict, info):
        input_dict["image_paths"] = []
        input_dict["lidar2camera"] = []
        input_dict["lidar2image"] = []
        input_dict["camera2ego"] = []
        input_dict["camera_intrinsics"] = []
        input_dict["camera2lidar"] = []

        for cam in self.cam_type:
            cam_info = info[cam]
            input_dict["image_paths"].append(info[cam])

            # lidar to camera transform
            lidar2camera_r = np.linalg.inv(np.array(self.calib_data[cam]["T_cam_lidar"])[:3, :3])
            lidar2camera_t = (
                np.array(self.calib_data[cam]["T_cam_lidar"])[3, :3] @ lidar2camera_r.T
            )
            lidar2camera_rt = np.eye(4).astype(np.float32)
            lidar2camera_rt[:3, :3] = lidar2camera_r.T
            lidar2camera_rt[3, :3] = -lidar2camera_t

            # print(f"pcdet解法:{lidar2camera_rt}")
            # print(f"求逆解法{np.linalg.inv(np.array(self.calib_data[cam]['T_cam_lidar']))}")
            
            input_dict["lidar2camera"].append(lidar2camera_rt.T)

            # camera intrinsics
            xi, fx, fy, cx, cy = self.calib_data[cam]["intrinsics"]
            camera_intrinsics_tmp=np.array([
                                [fx, 0, cx],
                                [0, fy, cy],
                                [0,  0,  1]
                            ])
            camera_intrinsics = np.eye(4).astype(np.float32)
            camera_intrinsics[:3, :3] = camera_intrinsics_tmp
            input_dict["camera_intrinsics"].append(camera_intrinsics)

            # lidar to image transform
            lidar2image = camera_intrinsics @ lidar2camera_rt.T
            input_dict["lidar2image"].append(lidar2image)

            # camera to ego transform
            # camera2ego = np.eye(4).astype(np.float32)
            # camera2ego[:3, :3] = Quaternion(
            #     camera_info["sensor2ego_rotation"]
            # ).rotation_matrix
            # camera2ego[:3, 3] = camera_info["sensor2ego_translation"]
            # input_dict["camera2ego"].append(camera2ego)
            


            # camera to lidar transform
            # camera2lidar = np.eye(4).astype(np.float32)
            # camera2lidar[:3, :3] = camera_info["sensor2lidar_rotation"]
            # camera2lidar[:3, 3] = camera_info["sensor2lidar_translation"]
            # input_dict["camera2lidar"].append(camera2lidar)

            camera2lidar=np.array(self.calib_data[cam]["T_cam_lidar"])
          
            camera2ego=camera2lidar.copy()
            # 没有lidar2ego，先是这样放着
            input_dict["camera2ego"].append(camera2ego)
            input_dict["camera2lidar"].append(camera2lidar)
        # read image
        filename = input_dict["image_paths"]
        images = []
        for name in filename:
            images.append(Image.open(str(self.root_path / name)))
        
        input_dict["camera_imgs"] = images
        input_dict["ori_shape"] = images[0].size
        
        # resize and crop image
        input_dict = self.crop_image(input_dict)

        return input_dict

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        
        self.split_file = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        for bag_name in self.split_file:
            folder_path=str(self.root_path)+"/"+bag_name + "/lidar0"
            all_names=os.listdir(folder_path)
            pcd_path = [folder_path+'/'+name for name in all_names if os.path.isfile(os.path.join(folder_path, name))]
            self.sample_id_list.extend(pcd_path)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.sample_id_list) * self.total_epochs

        return len(self.custom_infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)
        
        info = copy.deepcopy(self.custom_infos[index])
        sample_path = info['point_cloud']['lidar_path']
        points = self.get_lidar(sample_path)
        
        input_dict = {
            'frame_id': sample_path,
            'points': points
        }
        if self.use_camera:
            input_dict = self.load_camera_info(input_dict, info)
        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            gt_names = annos['name']
            gt_boxes_lidar = annos['gt_boxes_lidar']
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos, map_name_to_kitti):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
            )
            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos, self.map_class_to_kitti)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
    





    def get_infos(self, class_names, num_workers=1, has_label=True, sample_id_list=None, num_features=4):
        import concurrent.futures as futures

        def process_single_scene(lidar_path):
            print('%s sample_idx: %s' % (self.split, lidar_path))
            info = {}
            pc_info = {'num_features': num_features, 'lidar_path': lidar_path}
            cams_info={}
            bag_name=lidar_path.split("/")[-3]
            bag_path=str(self.root_path)+"/"+bag_name
            sample_id = lidar_path.split("/")[-1][:-4]
            info['point_cloud'] = pc_info
           
   
            if has_label:
                annotations = {}
                gt_boxes_lidar, name = self.get_label(sample_id,bag_path)
                annotations['name'] = name
                annotations['gt_boxes_lidar'] = gt_boxes_lidar[:, :7]
                info['annos'] = annotations

            for cam in self.cam_type:
                cam_path=str(self.root_path)+"/"+bag_name+"/"+cam+"/"+sample_id+".png"
                info[cam]=cam_path
            return info
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list

        # create a thread pool to improve the velocity
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    # def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
    #     import torch

    #     database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
    #     db_info_save_path = Path(self.root_path) / ('custom_dbinfos_%s.pkl' % split)

    #     database_save_path.mkdir(parents=True, exist_ok=True)
    #     all_db_infos = {}

    #     with open(info_path, 'rb') as f:
    #         infos = pickle.load(f)

    #     for k in range(len(infos)):
    #         print('gt_database sample: %d/%d' % (k + 1, len(infos)))
    #         info = infos[k]
    #         sample_idx = info['point_cloud']['lidar_idx']
    #         points = self.get_lidar(sample_idx)
    #         annos = info['annos']
    #         names = annos['name']
    #         gt_boxes = annos['gt_boxes_lidar']

    #         num_obj = gt_boxes.shape[0]
    #         point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
    #             torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
    #         ).numpy()  # (nboxes, npoints)

    #         for i in range(num_obj):
    #             filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
    #             filepath = database_save_path / filename
    #             gt_points = points[point_indices[i] > 0]

    #             gt_points[:, :3] -= gt_boxes[i, :3]
    #             with open(filepath, 'w') as f:
    #                 gt_points.tofile(f)

    #             if (used_classes is None) or names[i] in used_classes:
    #                 db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
    #                 db_info = {'name': names[i], 'path': db_path, 'gt_idx': i,
    #                            'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0]}
    #                 if names[i] in all_db_infos:
    #                     all_db_infos[names[i]].append(db_info)
    #                 else:
    #                     all_db_infos[names[i]] = [db_info]

    #     # Output the num of all classes in database
    #     for k, v in all_db_infos.items():
    #         print('Database %s: %d' % (k, len(v)))

    #     with open(db_info_save_path, 'wb') as f:
    #         pickle.dump(all_db_infos, f)

    @staticmethod
    def create_label_file_with_name_and_box(class_names, gt_names, gt_boxes, save_label_path):
        with open(save_label_path, 'w') as f:
            for idx in range(gt_boxes.shape[0]):
                boxes = gt_boxes[idx]
                name = gt_names[idx]
                if name not in class_names:
                    continue
                line = "{x} {y} {z} {l} {w} {h} {angle} {name}\n".format(
                    x=boxes[0], y=boxes[1], z=(boxes[2]), l=boxes[3],
                    w=boxes[4], h=boxes[5], angle=boxes[6], name=name
                )
                f.write(line)


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=1):
    dataset = CustomDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    num_features = len(dataset_cfg.POINT_FEATURE_ENCODING.src_feature_list)

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)

    print('------------------------Start to generate data infos------------------------')

    dataset.set_split(train_split)
    custom_infos_train = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(custom_infos_train, f)
    print('Custom info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    custom_infos_val = dataset.get_infos(
        class_names, num_workers=workers, has_label=True, num_features=num_features
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(custom_infos_val, f)
    print('Custom info train file is saved to %s' % val_filename)

    # print('------------------------Start create groundtruth database for data augmentation------------------------')
    # dataset.set_split(train_split)
    # dataset.create_groundtruth_database(train_filename, split=train_split)
    # print('------------------------Data preparation done------------------------')


if __name__ == '__main__':
    import sys

    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':


        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom',
        )
