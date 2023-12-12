import torch
import configargparse

from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *

import os, sys, copy
from nerf.adv.awp import AdvWeightPerturb

from functools import partial
from loss import huber_loss


if __name__ == '__main__':

    parser = configargparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--num_testval_images', type=int, default=None, help="how many validation/test images to use")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### table options
    parser.add_argument('--write_table', action='store_true')
    parser.add_argument('--implementation_name', type=str, default=None, help="name of the implementation")
    parser.add_argument('--dataset_name', type=str, default=None, help="name of the file where to output the table")

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=64, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=128, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")
    parser.add_argument('--few_shot', type=int, default=0, help="how many images to train (for few-shot setting)")

    ### Depth Smoothness regularization with patch_size=True
    parser.add_argument('--rgb_weighting', action='store_true', help="use color difference b/w gt and prediction as geometric loss weighting factor")
    parser.add_argument('--patch_gamma', type=int, default=1, help="gamma for geometric loss weighting factor")
    parser.add_argument('--depth_reg_lambda', type=float, default=1.0, help="lambda parameter for depth geometric regularization")
    ### Sample Space Annealing
    parser.add_argument('--anneal_nearfar', action='store_true', help="anneal near and far bound during initial training, useful to catch center of the scene in few-shot")
    parser.add_argument('--anneal_nearfar_steps', type=int, default=512, help="steps for near/far annealing")
    parser.add_argument('--anneal_nearfar_perc', type=float, default=0.2, help="percentage for near/far annealing")
    parser.add_argument('--anneal_mid_perc', type=float, default=0.5, help="percentage for near/far mid point")

    ### Frequency Regularization
    parser.add_argument('--freq_reg_mask_sigma', action='store_true',
                        help="use frequency regularization mask after input encoding in sigma network")
    parser.add_argument('--freq_reg_mask_color', action='store_true',
                        help="use frequency regularization mask after input encoding in color network")
    parser.add_argument('--total_iter_end_rate', type=float, default=0.9, help="set total iteration rate for encoding mask regularization")
    parser.add_argument('--start_ptr', type=int, default=1, help="how many feature embeddings we consider at the beginning")
    parser.add_argument('--num_levels', type=int, default=16, help="number of hash encoding levels for the density network")

    ### Lipshitz regularization
    parser.add_argument('--use_lipshitz_sigma', action='store_true', help="use lipshitz in sigma network")
    parser.add_argument('--use_lipshitz_color', action='store_true', help="use lipshitz in color network")

    ### Diffusion Geometric regularization
    parser.add_argument('--diff_reg', action='store_true', help="use diffusion geometric regulazition additional losses")
    parser.add_argument('--loss_dist', action='store_true', help="use loss_dist")
    parser.add_argument('--dist_lambda', type=float, default=0.001, help="lambda parameter for dist_loss term")
    parser.add_argument('--diff_reg_start_iter', type=int, default=2000, help="start diffusion geometric regularization after certain iter")
    parser.add_argument('--diff_reg_end_rate', type=float , default=1.0, help="stop diffusion geometric regularization after a ceratain iters percentage")
    parser.add_argument('--use_depth', action='store_true', help="use depth for computing the loss_dist")
    parser.add_argument('--diff_num_rays', type=int, default=250, help="how many rays to consider for the loss_dist computation")
    parser.add_argument('--loss_fg', action='store_true', help="use loss_fg")
    parser.add_argument('--fg_lambda', type=float, default=0.01, help="lambda parameter for fg_loss (sum of weight unity)")

    ### Adversarial training
    parser.add_argument("--adv", nargs='*', type=str, default=[],
                        help='turn on adv training. support combination of adv type')
    parser.add_argument("--unadv", action='store_true', default=False,
                        help='turn on unadv training')
    parser.add_argument("--adv_type", type=str, default='pgd', choices=['random', 'pgd'],
                        help='type of adv noises: random or pgd')
    parser.add_argument("--adv_lambda", type=float, default=0.5,
                        help='lambda coefficient of adv loss')
    parser.add_argument("--pgd_alpha", nargs='*', type=float, default=[1e-5],
                        help='alpha for pgd noise searching')
    parser.add_argument("--pgd_iters", nargs='*', type=int, default=[1],
                        help='iteration number for pgd noise searching')
    parser.add_argument("--pgd_eps", nargs='*', type=float, default=[1e-5],
                        help='maximal perturbation stength in ratio or magnitude')
    parser.add_argument("--pgd_norm", nargs='*', type=str, default=['l_inf'],
                        help='boundary in norm of pgd noise searching')
    parser.add_argument("--awp_warmup", type=int, default=0,
                        help='warm up iterations for awp')
    parser.add_argument("--awp_gamma", type=float, default=0.01,
                        help='gamma for awp training')
    parser.add_argument("--awp_lrate", type=float, default=5e-4,
                        help='lrate for proxy optimizer in awp training')

    ### Ray Entropy Minimization Loss
    # entropy
    parser.add_argument("--N_entropy", type=int, default=100,
                        help='number of entropy ray')
    # entropy type
    parser.add_argument("--entropy", action='store_true',
                        help='using entropy ray loss')
    parser.add_argument("--entropy_log_scaling", action='store_true',
                        help='using log scaling for entropy loss')
    parser.add_argument("--entropy_ignore_smoothing", action='store_true',
                        help='ignoring entropy for ray for smoothing')
    parser.add_argument("--entropy_end_iter", type=int, default=None,
                        help='end iteratio of entropy')
    parser.add_argument("--entropy_type", type=str, default='log2', choices=['log2', '1-p'],
                        help='choosing type of entropy')
    parser.add_argument("--entropy_acc_threshold", type=float, default=0.1,
                        help='threshold for acc masking')
    parser.add_argument("--computing_entropy_all", action='store_true',
                        help='computing entropy for both seen and unseen ')
    # lambda
    parser.add_argument("--entropy_ray_lambda", type=float, default=1,
                        help='entropy lambda for ray entropy loss')
    parser.add_argument("--entropy_ray_zvals_lambda", type=float, default=1,
                        help='entropy lambda for ray zvals entropy loss')


    ### Infomation Gain Reduction Loss (KL-Divergence loss)
    parser.add_argument("--smoothing", action='store_true',
                        help='using information gain reduction loss')
    # choosing between rotating camera pose & near pixel
    parser.add_argument("--smooth_sampling_method", type=str, default='near_pose',
                        help='how to sample the near rays, near_pose: modifying camera pose, near_pixel: sample near pixel',
                        choices=['near_pose', 'near_pixel'])
    # 1) sampling by rotating camera pose
    parser.add_argument("--near_c2w_type", type=str, default='rot_from_origin',
                        help='random augmentation method')
    parser.add_argument("--near_c2w_rot", type=float, default=5,
                        help='random augmentation rotate: degree')
    parser.add_argument("--near_c2w_trans", type=float, default=0.1,
                        help='random augmentation translation')
    # 2) sampling with near pixel
    parser.add_argument("--smooth_pixel_range", type=int, default=1,
                        help='the maximum distance between the near ray & the original ray (pixel dimension)')
    # optimizing
    parser.add_argument("--smoothing_lambda", type=float, default=0.001,
                        help='lambda for smoothing loss')
    parser.add_argument("--smoothing_activation", type=str, default='norm',
                        help='how to make alpha to the distribution')
    parser.add_argument("--smoothing_step_size", type=int, default='5000',
                        help='reducing smoothing every')
    parser.add_argument("--smoothing_rate", type=float, default=0.5,
                        help='reducing smoothing rate')
    parser.add_argument("--smoothing_end_iter", type=int, default=None,
                        help='when smoothing will be end')

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--aabb_box', type=float, nargs=6, default=None, help="aabb only used for generating points")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")
    parser.add_argument('--downscale', type=int, default=1, help="Set downscale for resolution of images")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")


    opt, _ = parser.parse_known_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
        device=device,
        num_levels=opt.num_levels,
        lipshitz_color=opt.use_lipshitz_color,
        lipshitz_sigma=opt.use_lipshitz_sigma,
        freq_reg_mask_color=opt.freq_reg_mask_color,
        freq_reg_mask_sigma=opt.freq_reg_mask_sigma,
        aabb_box=opt.aabb_box,
    )
    
    print(model)

    # construct adv weight perturber
    awp_adversary = None
    if 'awp' in opt.adv:
        proxy = copy.deepcopy(model)
        proxy_optim = torch.optim.Adam(params=proxy.parameters(), lr=opt.awp_lrate, betas=(0.9, 0.999))
        awp_adversary = AdvWeightPerturb(model, proxy, proxy_optim, opt.awp_gamma)

    criterion = torch.nn.MSELoss(reduction='none')
    #criterion = partial(huber_loss, reduction='none')
    #criterion = torch.nn.HuberLoss(reduction='none', beta=0.1) # only available after torch 1.10 ?
    
    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()

            #if test_loader.has_gt:
            trainer.evaluate(test_loader, type='test') # blender has gt, so evaluate it.
    
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)

        train_loader = NeRFDataset(opt, device=device, type='train', downscale=opt.downscale).dataloader()

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device), SSIMMeter(device=device)]
        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50, awp_adversary=awp_adversary)

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        
        else:
            valid_loader = NeRFDataset(opt, device=device, type='val', downscale=opt.downscale).dataloader()

            max_epoch = np.ceil(opt.iters / len(train_loader)).astype(np.int32)
            trainer.train(train_loader, valid_loader, max_epoch)

            # also test
            test_loader = NeRFDataset(opt, device=device, type='test', downscale=opt.downscale).dataloader()
            
            #if test_loader.has_gt:
            trainer.evaluate(test_loader, type='test') # blender has gt, so evaluate it.
            
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)