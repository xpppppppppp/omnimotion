import configargparse


def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')

    # general
    parser.add_argument('--data_dir', type=str, help='the directory for the video sequence')
    parser.add_argument('--expname', type=str, default='', help='experiment name')
    parser.add_argument('--local_rank', type=int, default=0, help='rank for distributed training')
    parser.add_argument('--save_dir', type=str, default='out/', help='output dir')
    parser.add_argument('--ckpt_path', type=str, default='', help='checkpoint path')
    parser.add_argument('--no_reload', action='store_true', help='do not reload the weights')
    parser.add_argument('--distributed', type=int, default=0, help='if use distributed training')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--load_opt', type=int, default=1, help='if loading optimizers')
    parser.add_argument('--load_scheduler', type=int, default=1, help='if loading schedulers')
    parser.add_argument('--loader_seed', type=int, default=12,
                        help='the random seed used for DataLoader')

    # data
    parser.add_argument('--dataset_types', type=str, default='flow', help='training datasets; flow is the supported training path')
    parser.add_argument('--dataset_weights', nargs='+', type=float, default=[1.], help='the weight for each dataset')
    parser.add_argument('--num_imgs', type=int, default=250, help='max number of images to train')
    parser.add_argument('--num_pairs', type=int, default=8, help='# image pairs to sample in each batch')
    parser.add_argument('--num_pts', type=int, default=256, help='# pts to sample from each pair of images')
    parser.add_argument('--keypoint_dir', type=str, default='', help='directory containing per-frame keypoint json files')
    parser.add_argument('--keypoint_format', type=str, default='auto', help='auto, labelme_points, openpose, keypoints_xy')
    parser.add_argument('--num_joints', type=int, default=17, help='number of joints expected in each keypoint file')
    parser.add_argument('--min_keypoint_conf', type=float, default=0.5, help='minimum confidence for a keypoint to participate in training or querying')
    parser.add_argument('--patch_size', type=int, default=5, help='local neighborhood size around each keypoint for training')
    parser.add_argument('--foreground_hard_mask', action='store_true', help='discard keypoint samples that fall outside the foreground mask')
    parser.add_argument('--query_pts_source', type=str, default='mask', help='mask or keypoints for visualization queries')
    parser.add_argument('--query_keypoint_path', type=str, default='', help='optional keypoint json file for the query frame only')
    parser.add_argument('--rolling_query', action='store_true', help='track points by propagating frame to frame instead of querying all frames from the source frame')

    # lr
    parser.add_argument('--lr_feature', type=float, default=1e-3, help='learning rate for feature mlp')
    parser.add_argument('--lr_deform', type=float, default=1e-4, help='learning rate for deform mlp')
    parser.add_argument('--lr_color', type=float, default=3e-4, help='learning rate for color mlp')
    parser.add_argument("--lrate_decay_steps", type=int, default=20000,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--lrate_decay_factor", type=float, default=0.5,
                        help='decay learning rate by a factor every specified number of steps')
    parser.add_argument("--grad_clip", type=float, default=0, help='clip the gradient to avoid training instability')

    # point tracking head
    parser.add_argument('--use_point_head', action='store_true',
                        help='train an explicit MLP head for sparse point/keypoint tracking')
    parser.add_argument('--use_point_head_for_query', action='store_true',
                        help='use the point tracking head instead of the OmniMotion ray compositor during query/viz')
    parser.add_argument('--lr_point', type=float, default=1e-4, help='learning rate for point tracking head')
    parser.add_argument('--point_head_hidden', type=int, default=256, help='hidden size for point tracking head')
    parser.add_argument('--point_head_layers', type=int, default=3, help='number of hidden layers in point tracking head')
    parser.add_argument('--point_head_residual', action='store_true', default=True,
                        help='predict a residual correction on top of OmniMotion correspondence')
    parser.add_argument('--no_point_head_residual', dest='point_head_residual', action='store_false',
                        help='make point head predict from the source point directly')
    parser.add_argument('--point_delta_scale', type=float, default=0.25,
                        help='maximum normalized residual displacement predicted by the point head')
    parser.add_argument('--point_use_rgb_patch', action='store_true',
                        help='concatenate query/base RGB patch features into the point head')
    parser.add_argument('--point_rgb_patch_size', type=int, default=5,
                        help='odd patch size sampled around source point and base target point')
    parser.add_argument('--point_use_dino_feature', action='store_true',
                        help='concatenate query/base DINO point features into the point head')
    parser.add_argument('--point_dino_dir', type=str, default='',
                        help='directory containing per-frame DINO .npy feature grids')
    parser.add_argument('--point_dino_dim', type=int, default=384,
                        help='DINO feature dimension used by the point head')
    parser.add_argument('--point_dino_l2_normalize', action='store_true',
                        help='L2-normalize sampled DINO features before concatenating them into the point head')
    parser.add_argument('--point_use_dino_correlation', action='store_true',
                        help='use a local DINO cosine-correlation heatmap around the base target point')
    parser.add_argument('--point_corr_radius', type=int, default=12,
                        help='pixel radius for the local DINO correlation search window')
    parser.add_argument('--point_corr_stride', type=int, default=2,
                        help='pixel stride for the local DINO correlation search window')
    parser.add_argument('--point_corr_temperature', type=float, default=10.0,
                        help='softmax temperature used for local DINO correlation soft-argmax')
    parser.add_argument('--point_corr_update_base', action='store_true',
                        help='use local DINO correlation soft-argmax as the residual base point')
    parser.add_argument('--point_loss_weight', type=float, default=1.0, help='weight for supervised keypoint tracking loss')
    parser.add_argument('--point_conf_weight', type=float, default=0.01, help='weight for point visibility/confidence loss')
    parser.add_argument('--point_num_pairs', type=int, default=8, help='number of keypoint frame pairs sampled per step')
    parser.add_argument('--point_max_interval', type=int, default=0,
                        help='maximum frame interval for point head supervision; 0 samples any pair')
    parser.add_argument('--point_supervision', type=str, default='flow',
                        help='flow uses RAFT batch pts1/pts2; keypoints uses keypoint_dir frame pairs')

    # network training
    parser.add_argument('--use_error_map', action='store_true', help='use error map')
    parser.add_argument('--use_count_map', action='store_true', help='use count map')
    parser.add_argument('--train_use_mask', action='store_true',
                        help='restrict flow supervision sampling to foreground masks in source and target frames')
    parser.add_argument('--train_mask_erosion', type=int, default=0,
                        help='optional erosion kernel size applied to training foreground masks before sampling')
    parser.add_argument('--train_keypoint_bias', action='store_true',
                        help='prefer flow supervision near rolling keypoint tracks during training')
    parser.add_argument('--train_keypoint_track_dir', type=str, default='',
                        help='directory containing per-frame pseudo keypoint json files used to bias training sampling')
    parser.add_argument('--train_keypoint_radius', type=int, default=24,
                        help='radius in pixels for per-frame keypoint neighborhoods used in training')
    parser.add_argument('--train_keypoint_focus_ratio', type=float, default=0.75,
                        help='fraction of sampled flow points drawn from keypoint neighborhoods when available')
    parser.add_argument('--train_query_frame_only', action='store_true',
                        help='always use query_frame_id as the source frame during training; only meaningful for static first-frame bias')
    parser.add_argument('--train_query_frame_prob', type=float, default=0.5,
                        help='probability of choosing query_frame_id as the source frame when only static first-frame keypoint bias is used')
    parser.add_argument('--use_affine', action='store_true',
                        help='if using additional 2D affine transformation layers for x, y in the invertible network')
    parser.add_argument('--mask_near', action='store_true',
                        help='if mask out the nearest samples in the beginning of the optimization,'
                             'may be helpful to avoid bad initialization associated with wrong surface ordering'
                             'e.g., a surface is initialized at very small depth but should instead be farther away')
    parser.add_argument('--num_samples_ray', type=int, default=32, help='number of samples per ray')
    parser.add_argument('--pe_freq', type=int, default=4, help='the freq for pe used in the affine coupling layers')
    parser.add_argument('--min_depth', type=float, default=0, help='the minimum depth value')
    parser.add_argument('--max_depth', type=float, default=2, help='the maximum depth value')
    parser.add_argument('--start_interval', type=int, default=20, help='the starting interval')
    parser.add_argument('--max_padding', type=float, default=0,
                        help='if predicted pixel locs exceed this padding, mask them out for training')

    # inference
    parser.add_argument('--chunk_size', type=int, default=10000, help='chunk size for rendering depth and rgb')
    parser.add_argument('--use_max_loc', action='store_true',
                        help='during inference, if using only the sample with maximum blending weight on the ray'
                             'to compute correspondence. If set to False, the correspondences will be computed'
                             'the same way as training, i.e., compositing all samples along the ray.')
    parser.add_argument('--query_frame_id', type=int, default=0, help='the id of the query frame')
    parser.add_argument('--vis_occlusion', action='store_true',
                        help='if marking occluded pixels as crosses for visualization')
    parser.add_argument('--occlusion_th', type=float, default=0.99,
                        help='to determine if a mapped 3d location in the target frame is occluded or not,'
                             ' we look at the fraction of light absorbed by samples in front of this location '
                             'on the ray in the target frame (i.e., 1 - transmittance)'
                             'if that value is higher than this threshold, the mapped point is considered as occluded')
    parser.add_argument('--foreground_mask_path', type=str, default='',
                        help='providing the path for foreground mask file for generating trails')

    # log
    parser.add_argument('--i_print', type=int, default=100, help='frequency for printing losses')
    parser.add_argument('--i_img', type=int, default=500, help='frequency for writing visualizations to tensorboard')
    parser.add_argument('--i_weight', type=int, default=10000, help='frequency for saving ckpts')
    parser.add_argument('--i_cache', type=int, default=10000, help='frequency for caching current flow predictions')
    parser.add_argument('--skip_checkpoint_visualization', action='store_true',
                        help='save checkpoint weights without rendering checkpoint visualization videos')

    parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

    args = parser.parse_args()
    return args
