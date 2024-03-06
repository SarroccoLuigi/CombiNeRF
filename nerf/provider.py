import os
import cv2
import glob
import json
#from cv2 import transform
import tqdm
import numpy as np
from scipy.spatial.transform import Slerp, Rotation

import trimesh

import torch
from torch.utils.data import DataLoader

from .utils import get_rays
from nerf.info.generate_near_c2w import GetNearC2W, get_near_pixel


# ref: https://github.com/NVlabs/instant-ngp/blob/b76004c8cf478880227401ae763be4c02f80b62f/include/neural-graphics-primitives/nerf_loader.h#L50
def nerf_matrix_to_ngp(pose, scale=0.33, offset=[0, 0, 0]):
    # for the fox dataset, 0.33 scales camera radius to ~ 2
    new_pose = np.array([
        [pose[1, 0], -pose[1, 1], -pose[1, 2], pose[1, 3] * scale + offset[0]],
        [pose[2, 0], -pose[2, 1], -pose[2, 2], pose[2, 3] * scale + offset[1]],
        [pose[0, 0], -pose[0, 1], -pose[0, 2], pose[0, 3] * scale + offset[2]],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    return new_pose


def visualize_poses(poses, size=0.1):
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def rand_poses(size, device, radius=1, theta_range=[np.pi/3, 2*np.pi/3], phi_range=[0, 2*np.pi]):
    ''' generate random poses from an orbit camera
    Args:
        size: batch size of generated poses.
        device: where to allocate the output.
        radius: camera radius
        theta_range: [min, max], should be in [0, \pi]
        phi_range: [min, max], should be in [0, 2\pi]
    Return:
        poses: [size, 4, 4]
    '''
    
    def normalize(vectors):
        return vectors / (torch.norm(vectors, dim=-1, keepdim=True) + 1e-10)

    thetas = torch.rand(size, device=device) * (theta_range[1] - theta_range[0]) + theta_range[0]
    phis = torch.rand(size, device=device) * (phi_range[1] - phi_range[0]) + phi_range[0]

    centers = torch.stack([
        radius * torch.sin(thetas) * torch.sin(phis),
        radius * torch.cos(thetas),
        radius * torch.sin(thetas) * torch.cos(phis),
    ], dim=-1) # [B, 3]

    # lookat
    forward_vector = - normalize(centers)
    up_vector = torch.FloatTensor([0, -1, 0]).to(device).unsqueeze(0).repeat(size, 1) # confused at the coordinate system...
    right_vector = normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float, device=device).unsqueeze(0).repeat(size, 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = centers

    return poses


class NeRFDataset:
    def __init__(self, opt, device, type='train', downscale=1, n_test=10):
        super().__init__()
        
        self.opt = opt
        self.device = device
        self.type = type # train, val, test
        self.downscale = downscale
        self.root_path = opt.path
        self.preload = opt.preload # preload data into GPU
        self.scale = opt.scale # camera radius scale to make sure camera are inside the bounding box.
        self.offset = opt.offset # camera offset
        self.bound = opt.bound # bounding box half length, also used as the radius to random sample poses.
        self.fp16 = opt.fp16 # if preload, load into fp16.

        self.training = self.type in ['train', 'all', 'trainval']
        self.num_rays = self.opt.num_rays if self.training else -1

        self.rand_pose = opt.rand_pose

        if opt.smoothing:
            self.get_near_c2w = GetNearC2W(opt, device=self.device)

        # auto-detect transforms.json and split mode.
        if os.path.exists(os.path.join(self.root_path, 'transforms.json')):
            self.mode = 'colmap' # manually split, use view-interpolation for test.
        elif os.path.exists(os.path.join(self.root_path, 'transforms_train.json')):
            self.mode = 'blender' # provided split
        else:
            raise NotImplementedError(f'[NeRFDataset] Cannot find transforms*.json under {self.root_path}')

        # load nerf-compatible format data.
        if self.mode == 'colmap':
            with open(os.path.join(self.root_path, 'transforms.json'), 'r') as f:
                transform = json.load(f)
        elif self.mode == 'blender':
            # load all splits (train/valid/test), this is what instant-ngp in fact does...
            if type == 'all':
                transform_paths = glob.glob(os.path.join(self.root_path, '*.json'))
                transform = None
                for transform_path in transform_paths:
                    with open(transform_path, 'r') as f:
                        tmp_transform = json.load(f)
                        if transform is None:
                            transform = tmp_transform
                        else:
                            transform['frames'].extend(tmp_transform['frames'])
            # load train and val split
            elif type == 'trainval':
                with open(os.path.join(self.root_path, f'transforms_train.json'), 'r') as f:
                    transform = json.load(f)
                with open(os.path.join(self.root_path, f'transforms_val.json'), 'r') as f:
                    transform_val = json.load(f)
                transform['frames'].extend(transform_val['frames'])
            # only load one specified split
            else:
                with open(os.path.join(self.root_path, f'transforms_{type}.json'), 'r') as f:
                    transform = json.load(f)

        else:
            raise NotImplementedError(f'unknown dataset mode: {self.mode}')

        # load image size
        if 'h' in transform and 'w' in transform:
            self.H = int(transform['h']) // downscale
            self.W = int(transform['w']) // downscale
        else:
            # we have to actually read an image to get H and W later.
            self.H = self.W = None
        
        # read images
        frames = transform["frames"]
        #frames = sorted(frames, key=lambda d: d['file_path']) # why do I sort...
        
        # for colmap, manually interpolate a test set.
        """"
        if self.mode == 'colmap' and type == 'test':
            
            # choose two random poses, and interpolate between.
            f0, f1 = np.random.choice(frames, 2, replace=False)
            pose0 = nerf_matrix_to_ngp(np.array(f0['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            pose1 = nerf_matrix_to_ngp(np.array(f1['transform_matrix'], dtype=np.float32), scale=self.scale, offset=self.offset) # [4, 4]
            rots = Rotation.from_matrix(np.stack([pose0[:3, :3], pose1[:3, :3]]))
            slerp = Slerp([0, 1], rots)

            self.poses = []
            self.images = None
            for i in range(n_test + 1):
                ratio = np.sin(((i / n_test) - 0.5) * np.pi) * 0.5 + 0.5
                pose = np.eye(4, dtype=np.float32)
                pose[:3, :3] = slerp(ratio).as_matrix()
                pose[:3, 3] = (1 - ratio) * pose0[:3, 3] + ratio * pose1[:3, 3]
                self.poses.append(pose)
        """

        #else:
        # for colmap, manually split a valid set (the first frame). Set for fox dataset
        if self.mode == 'colmap':
            if type == 'train':
                frames = frames[5:]#frames[0:14]
            elif type == 'val':
                frames = frames[:1]#frames[14:16]
            # else 'all' or 'trainval' : use all frames
            elif type == 'test':
                frames = frames[1:5]


        self.poses = []
        self.images = []
        check_gray = False
        for f in tqdm.tqdm(frames, desc=f'Loading {type} data'):
            f_path = os.path.join(self.root_path, f['file_path'])
            if self.mode == 'blender' and '.' not in os.path.basename(f_path):
                f_path += '.png' # so silly...

            # there are non-exist paths in fox...
            if not os.path.exists(f_path):
                continue

            pose = np.array(f['transform_matrix'], dtype=np.float32) # [4, 4]
            pose = nerf_matrix_to_ngp(pose, scale=self.scale, offset=self.offset)

            image = cv2.imread(f_path, cv2.IMREAD_UNCHANGED) # [H, W, 3] o [H, W, 4]
            if self.H is None or self.W is None:
                self.H = image.shape[0] // downscale
                self.W = image.shape[1] // downscale

            # add support for the alpha channel as a mask.
            if len(image.shape) == 2:
                check_gray = True
            if image.shape[-1] == 3:
                if check_gray:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    for i in range(1536):
                        for j in range(2048):
                            image[i,j] = int(image[i,j]/1.8)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)

            if image.shape[0] != self.H or image.shape[1] != self.W:
                image = cv2.resize(image, (self.W, self.H), interpolation=cv2.INTER_AREA)

            image = image.astype(np.float32) / 255 # [H, W, 3/4]

            self.poses.append(pose)
            self.images.append(image)


        # few shot
        if type == 'train' and self.opt.few_shot > 0 and self.opt.few_shot < len(self.images):
            if self.opt.few_shot == 8: # NeRF-Synthetic
                indicies = [26, 86, 2, 55, 75, 93, 16, 73]
                self.images = [self.images[i] for i in indicies]
                self.poses = [self.poses[i] for i in indicies]
            else:
                idx_sub = np.linspace(0, len(self.images) - 1, self.opt.few_shot)
                idx_sub = [round(i) for i in idx_sub]
                self.images = [self.images[i] for i in idx_sub]
                self.poses = [self.poses[i] for i in idx_sub]

        if (type == 'test' or type == 'val') and self.opt.num_testval_images is not None:
            idx_sub = np.linspace(0, len(self.images) - 1, self.opt.num_testval_images)
            idx_sub = [round(i) for i in idx_sub]
            self.images = [self.images[i] for i in idx_sub]
            self.poses = [self.poses[i] for i in idx_sub]

            
        self.poses = torch.from_numpy(np.stack(self.poses, axis=0)) # [N, 4, 4]
        if self.images is not None:
            self.images = torch.from_numpy(np.stack(self.images, axis=0)) # [N, H, W, C]
        
        # calculate mean radius of all camera poses
        self.radius = self.poses[:, :3, 3].norm(dim=-1).mean(0).item()
        #print(f'[INFO] dataset camera poses: radius = {self.radius:.4f}, bound = {self.bound}')

        # initialize error_map
        if self.training and self.opt.error_map:
            self.error_map = torch.ones([self.images.shape[0], 128 * 128], dtype=torch.float) # [B, 128 * 128], flattened for easy indexing, fixed resolution...
        else:
            self.error_map = None

        # [debug] uncomment to view all training poses.
        # visualize_poses(self.poses.numpy())

        # [debug] uncomment to view examples of randomly generated poses.
        # visualize_poses(rand_poses(100, self.device, radius=self.radius).cpu().numpy())

        if self.preload:
            self.poses = self.poses.to(self.device)
            if self.images is not None:
                # TODO: linear use pow, but pow for half is only available for torch >= 1.10 ?
                if self.fp16 and self.opt.color_space != 'linear':
                    dtype = torch.half
                else:
                    dtype = torch.float
                self.images = self.images.to(dtype).to(self.device)
            if self.error_map is not None:
                self.error_map = self.error_map.to(self.device)

        # load intrinsics
        if 'fl_x' in transform or 'fl_y' in transform:
            fl_x = (transform['fl_x'] if 'fl_x' in transform else transform['fl_y']) / downscale
            fl_y = (transform['fl_y'] if 'fl_y' in transform else transform['fl_x']) / downscale
        elif 'camera_angle_x' in transform or 'camera_angle_y' in transform:
            # blender, assert in radians. already downscaled since we use H/W
            fl_x = self.W / (2 * np.tan(transform['camera_angle_x'] / 2)) if 'camera_angle_x' in transform else None
            fl_y = self.H / (2 * np.tan(transform['camera_angle_y'] / 2)) if 'camera_angle_y' in transform else None
            if fl_x is None: fl_x = fl_y
            if fl_y is None: fl_y = fl_x
        else:
            raise RuntimeError('Failed to load focal length, please check the transforms.json!')

        cx = (transform['cx'] / downscale) if 'cx' in transform else (self.W / 2)
        cy = (transform['cy'] / downscale) if 'cy' in transform else (self.H / 2)

        self.intrinsics = np.array([fl_x, fl_y, cx, cy])


    def collate(self, index):

        B = len(index) # a list of length 1

        # random pose without gt images.
        if self.rand_pose == 0 or index[0] >= len(self.poses):

            poses = rand_poses(B, self.device, radius=self.radius)

            # sample a low-resolution but full image for CLIP
            s = np.sqrt(self.H * self.W / self.num_rays) # only in training, assert num_rays > 0
            rH, rW = int(self.H / s), int(self.W / s)
            rays = get_rays(poses, self.intrinsics / s, rH, rW, -1)

            return {
                'H': rH,
                'W': rW,
                'rays_o': rays['rays_o'],
                'rays_d': rays['rays_d'],    
            }

        poses = self.poses[index].to(self.device) # [B, 4, 4]

        error_map = None if self.error_map is None else self.error_map[index]

        if self.type == 'test':
            rays = get_rays(poses, self.intrinsics, self.H, self.W, -1)
        else:
            rays = get_rays(poses, self.intrinsics, self.H, self.W, self.num_rays, error_map, self.opt.patch_size)

        #rays = self.adjust_rays_to_ndc(rays, self.intrinsics[0], self.W, self.H)

        results = {
            'H': self.H,
            'W': self.W,
            'rays_o': rays['rays_o'],
            'rays_d': rays['rays_d'],
        }

        if self.images is not None:
            images = self.images[index].to(self.device) # [B, H, W, 3/4]
            if self.training:
                C = images.shape[-1]
                images = torch.gather(images.view(B, -1, C), 1, torch.stack(C * [rays['inds']], -1)) # [B, N, 3/4]
            results['images'] = images
            results['cam_id'] = index
        
        # need inds to update error_map
        if error_map is not None:
            results['index'] = index
            results['inds_coarse'] = rays['inds_coarse']

        # entropy and gain loss
        if self.type == 'train':
            if self.opt.smoothing:
                for pose in poses: # only one pose
                    if self.opt.smooth_sampling_method == 'near_pixel':
                        pose = pose[..., None, :, :]
                        near_inds = get_near_pixel(rays['inds'], self.opt.smooth_pixel_range, self.H, self.W, self.device)
                        near_rays = get_rays(pose, self.intrinsics, self.H, self.W, self.num_rays, patch_size=self.opt.patch_size, ray_inds=near_inds)
                    else:
                        near_pose = self.get_near_c2w(pose[:3, :4]) # take pose without [0,0,0,1]
                        near_pose = torch.cat([near_pose[None, ...], pose[None, 3:4, :4]], 1) # necessary for tensor dimensionality in get_rays
                        near_rays = get_rays(near_pose, self.intrinsics, self.H, self.W, self.num_rays, patch_size=self.opt.patch_size, ray_inds=rays['inds'])

            if self.opt.entropy and (self.opt.N_entropy != 0):
                rays_entropy = get_rays(poses, self.intrinsics, self.H, self.W, self.opt.N_entropy, patch_size=self.opt.patch_size)
                if self.opt.smoothing:
                    for pose in poses:
                        if self.opt.smooth_sampling_method == 'near_pixel':
                            pose = pose[..., None, :, :]
                            near_inds = get_near_pixel(rays_entropy['inds'], self.opt.smooth_pixel_range, self.H, self.W,
                                                       self.device)
                            ent_near_rays = get_rays(pose, self.intrinsics, self.H, self.W, self.opt.N_entropy, patch_size=self.opt.patch_size, ray_inds=near_inds)
                        else:
                            ent_near_pose = self.get_near_c2w(pose[:3, :4])
                            ent_near_pose = torch.cat([ent_near_pose[None, ...], pose[None, 3:4, :4]], 1)
                            ent_near_rays = get_rays(ent_near_pose, self.intrinsics, self.H, self.W, self.opt.N_entropy, patch_size=self.opt.patch_size, ray_inds=rays_entropy['inds'])

            if self.opt.entropy and (self.opt.N_entropy != 0):
                results['rays_o'] = torch.cat([results['rays_o'], rays_entropy['rays_o']], 1)
                results['rays_d'] = torch.cat([results['rays_d'], rays_entropy['rays_d']], 1)

            if self.opt.smoothing:
                if self.opt.entropy and (self.opt.N_entropy != 0):
                    results['rays_o'] = torch.cat([results['rays_o'], near_rays['rays_o']], 1)
                    results['rays_d'] = torch.cat([results['rays_d'], near_rays['rays_d']], 1)
                    results['rays_o'] = torch.cat([results['rays_o'], ent_near_rays['rays_o']], 1)
                    results['rays_d'] = torch.cat([results['rays_d'], ent_near_rays['rays_d']], 1)
                else:
                    results['rays_o'] = torch.cat([results['rays_o'], near_rays['rays_o']], 1)
                    results['rays_d'] = torch.cat([results['rays_d'], near_rays['rays_d']], 1)

        return results

    def dataloader(self):
        size = len(self.poses)
        if self.training and self.rand_pose > 0:
            size += size // self.rand_pose # index >= size means we use random pose.
        loader = DataLoader(list(range(size)), batch_size=1, collate_fn=self.collate, shuffle=self.training, num_workers=0)
        loader._data = self # an ugly fix... we need to access error_map & poses in trainer.
        loader.has_gt = self.images is not None
        return loader

    def convert_to_ndc(self, origins, directions, focal, width, height, near=1.,
                       focaly=None):
        """Convert a set of rays to normalized device coordinates (NDC).

        Args:
          origins: np.ndarray(float32), [..., 3], world space ray origins.
          directions: np.ndarray(float32), [..., 3], world space ray directions.
          focal: float, focal length.
          width: int, image width in pixels.
          height: int, image height in pixels.
          near: float, near plane along the negative z axis.
          focaly: float, Focal for y axis (if None, equal to focal).

        Returns:
          origins_ndc: np.ndarray(float32), [..., 3].
          directions_ndc: np.ndarray(float32), [..., 3].

        This function assumes input rays should be mapped into the NDC space for a
        perspective projection pinhole camera, with identity extrinsic matrix (pose)
        and intrinsic parameters defined by inputs focal, width, and height.

        The near value specifies the near plane of the frustum, and the far plane is
        assumed to be infinity.

        The ray bundle for the identity pose camera will be remapped to parallel rays
        within the (-1, -1, -1) to (1, 1, 1) cube. Any other ray in the original
        world space can be remapped as long as it has dz < 0; this allows us to share
        a common NDC space for "forward facing" scenes.

        Note that
            projection(origins + t * directions)
        will NOT be equal to
            origins_ndc + t * directions_ndc
        and that the directions_ndc are not unit length. Rather, directions_ndc is
        defined such that the valid near and far planes in NDC will be 0 and 1.

        See Appendix C in https://arxiv.org/abs/2003.08934 for additional details.
        """

        directions = torch.Tensor.cpu(directions[0]).numpy()
        origins = torch.Tensor.cpu(origins[0]).numpy()

        # Shift ray origins to near plane, such that oz = -near.
        # This makes the new near bound equal to 0.
        t = -(near + origins[Ellipsis, 2]) / directions[Ellipsis, 2]
        origins = origins + t[Ellipsis, None] * directions

        dx, dy, dz = np.moveaxis(directions, -1, 0)
        ox, oy, oz = np.moveaxis(origins, -1, 0)

        fx = focal
        fy = focaly if (focaly is not None) else focal

        # Perspective projection into NDC for the t = 0 near points
        #     origins + 0 * directions
        origins_ndc = np.stack([
            -2. * fx / width * ox / oz, -2. * fy / height * oy / oz,
            -np.ones_like(oz)
        ],
            axis=-1)

        # Perspective projection into NDC for the t = infinity far points
        #     origins + infinity * directions
        infinity_ndc = np.stack([
            -2. * fx / width * dx / dz, -2. * fy / height * dy / dz,
            np.ones_like(oz)
        ],
            axis=-1)

        # directions_ndc points from origins_ndc to infinity_ndc
        directions_ndc = infinity_ndc - origins_ndc

        return origins_ndc, directions_ndc


    def adjust_rays_to_ndc(self, rays, focal, width, height):
        ndc_origins, ndc_directions = self.convert_to_ndc(rays['rays_o'],
                                                   rays['rays_d'],
                                                   focal, width, height)

        """"
        mat = ndc_origins
        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.linalg.norm(mat[:, :-1, :, :] - mat[:, 1:, :, :], axis=-1)
        dx = np.concatenate([dx, dx[:, -2:-1, :]], axis=1)
        dy = np.linalg.norm(mat[:, :, :-1, :] - mat[:, :, 1:, :], axis=-1)
        dy = np.concatenate([dy, dy[:, :, -2:-1]], axis=2)
        # Cut the distance in half, multiply it to match the variance of a uniform
        # distribution the size of a pixel (1/12, see paper).
        radii = (0.5 * (dx + dy))[Ellipsis, None] * 2 / np.sqrt(12)
        ones = np.ones_like(ndc_origins[Ellipsis, :1])

        rays = utils.Rays(
          origins=ndc_origins,
          directions=ndc_directions,
          viewdirs=rays.directions,
          radii=radii,
          lossmult=ones,
          near=ones * self.near,
          far=ones * self.far)
        """
        result={}
        ndc_origins=ndc_origins[None, ...]
        ndc_directions=ndc_directions[None, ...]

        result['rays_o']=torch.from_numpy(ndc_origins).to(device=self.device)
        result['rays_d']=torch.from_numpy(ndc_directions).to(device=self.device)
        if 'inds' in rays:
            result['inds']=rays['inds']

        return result