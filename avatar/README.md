# Creating an avatar from a phone scan

* This branch includes **avatar creation pipeline and animation function**.


## Directory
```
${ROOT}
|-- main
|-- common
|-- |-- utils/human_model_files
|-- |-- |-- smplx/SMPLX_FEMALE.npz
|-- |-- |-- smplx/SMPLX_MALE.npz
|-- |-- |-- smplx/SMPLX_NEUTRAL.npz
|-- |-- |-- smplx/MANO_SMPLX_vertex_ids.pkl
|-- |-- |-- smplx/SMPL-X__FLAME_vertex_ids.npy
|-- |-- |-- smplx/smplx_flip_correspondences.npz
|-- |-- |-- flame/flame_dynamic_embedding.npy
|-- |-- |-- flame/FLAME_FEMALE.pkl
|-- |-- |-- flame/FLAME_MALE.pkl
|-- |-- |-- flame/FLAME_NEUTRAL.pkl
|-- |-- |-- flame/flame_static_embedding.pkl
|-- |-- |-- flame/FLAME_texture.npz
|-- data
|-- |-- Custom
|-- |-- |-- data
|-- |-- NeuMan
|-- |-- |-- data/bike
|-- |-- |-- data/citron
|-- |-- |-- data/jogging
|-- |-- |-- data/lab
|-- |-- |-- data/parkinglot
|-- |-- |-- data/seattle
|-- tools
|-- output
```
* `main` contains high-level code for the avatar creation/animation and configurations.
* `common` contains kernel code. Download SMPL-X 1.1 version from [here](https://smpl-x.is.tue.mpg.de/download.php). Download FLAME 2020 version from [here](https://flame.is.tue.mpg.de/download.php).
* `data` contains data loading code.
* `tools` contains pre-processing and evaluation code.
* `output` contains log, visualized outputs, and fitting result.
* We use a modified 3DGS, which supports depth map and mask rendering. This is exactly the same as the original 3DGS except additional supports of the depth map and mask rendering. Please install the modified 3DGS from [here](https://github.com/leo-frank/diff-gaussian-rasterization-depth).

## Preprocessing of custom videos (your own video)
* We recommend capturing your own video outdoor as inside usually has too strong illuminations, which cast strong shadows. The less strong shadow, the better.
* Place your video at `data/Custom/data/$SUBJECT_ID/video.mp4`.
* Get optimized and smoothed SMPL-X parameters with [here](../fitting/).
* Go to `segment-anything` folder in `../fitting/tools` and run `python run_sam.py --root_path ../../data/Custom/data/$SUBJECT_ID` to obtain foreground masks.
* (Background option 1) If background of your own video is static, we get background point cloud with monocular depth estimator. To this end, go to `Depth-Anything-V2` folder in `../fitting/tools` and run `python run_depth_anything.py --root_path ../../data/Custom/data/$SUBJECT_ID`, which outputs `bkg_point_cloud.txt` at `data/Custom/data/$SUBJECT_ID`.
* (Background option 2) If background of your own video is dynamic (like NeuMan videos), go to `../fitting/tools/COLMAP` and run `python run_colmap.py --root_path ../../data/Custom/data/$SUBJECT_ID`. It will output `sparse` folder at `data/Custom/data/$SUBJECT_ID`.
* Prepare `frame_list_train.txt` and `frame_list_test.txt` in the same way as `frame_list_all.txt` of [here](../fitting/). Each will be used for training and evaluating the avatar, respectively. We recommend making them in 5 fps (assuming the original video and `frame_list_all.txt` are in 30 fps) to make the avatar creation faster.
* We provide an example in [here](https://drive.google.com/drive/folders/1e8RtE_eq_BitKwjx3iU1Ha5jfvdzHJNh?usp=sharing).

## Preprocessing of NeuMan videos
* You can download original NeuMan data from [here](https://github.com/apple/ml-neuman).
* We provide pre-processed Neuman data in [here](https://drive.google.com/drive/folders/15-V9EG21hT4pVhuBdHY3-lpvKjCuHbEU?usp=sharing).
* We provide train/test/validation split files in [here](https://drive.google.com/drive/folders/1L5KC4QIRX_ljQ_vyrIXV11FgynnuCDb8?usp=sharing), made following [the official code](https://github.com/apple/ml-neuman/blob/0149d258b2afe6ef65c91557bba9f874675871e4/data_io/neuman_helper.py#L149).
* We used the same pre-processing stage of the above one for the custom videos.

## Train
* Set `dataset` in `main/config.py`.
* Go to `main` folder and run `python train.py --subject_id $SUBJECT_ID`. The checkpoints are saved in `output/model/$SUBJECT_ID`.
* You can see reconstruction results on the training frames by running `python test.py --subject_id $SUBJECT_ID --test_epoch 4`. The results are saved to `output/result/$SUBJECT_ID`.

## Visualize a rotating avatar with the neutral pose
* Set `dataset` in `main/config.py`.
* Go to `main` folder and run `python get_neutral_pose.py --subject_id $SUBJECT_ID --test_epoch 4`.
* You can see a rotating avatar with the neutral pose in `./main/neutral_pose`.

## Animation
* Set `dataset` in `main/config.py`.
* Go to `main` folder and run `python animation.py --subject_id $SUBJECT_ID --test_epoch 4 --motion_path $PATH` if you want to use an avatar in `output/model_dump/$SUBJECT_ID`. `$PATH` should contain SMPL-X parameters to animate the avatar. You can prepare `$PATH` with [here](../fitting).
* To render the avatar from rotating camera, run `python animate_rot_cam.py --subject_id $SUBJECT_ID --test_epoch 4 --motion_path $PATH`.
* We provide SMPL-X parameters of several videos (examples of `$PATH`) in [here](https://drive.google.com/drive/folders/1ApDtoyqrcP2r2ZvX24eptmSefJvw_no5?usp=sharing).

## Test and evaluation (NeuMan dataset)
* For the evaluation on the NeuMan dataset, we optimize SMPL-X paraemeters of testing frames with image loss while fixing the pre-trained avatars following [1](https://github.com/aipixel/GaussianAvatar/issues/14), [2](https://github.com/mikeqzy/3dgs-avatar-release/issues/21), and [Section 4 B Evaluation](https://arxiv.org/pdf/2106.13629).
* Go to `tools` folder and run `python prepare_fit_pose_to_test.py --root_path ../output/model_dump/$SUBJECT_ID` if you want to use an avatar in `output/model_dump/$SUBJECT_ID`. It simply sets `epoch` of a checkpoint to 0 and save it to `'output/model_dump/$SUBJECT_ID' + '_fit_pose_to_test'`.
* Set `dataset='NeuMan'` in `main/config.py`.
* Go to `main` folder and run `python train.py --subject_id $SUBJECT_ID --fit_pose_to_test --continue`.
* You can see test results on the testing frames by running `python test.py --subject_id $SUBJECT_ID --fit_pose_to_test --test_epoch 4`. The results are saved to `'output/result/$SUBJECT_ID' + '_fit_pose_to_test'`.
* For the evaluation of the NeuMan dataset, go to `tools` folder and run `python eval_neuman.py --output_path '../output/result/$SUBJECT_ID' + '_fit_pose_to_test' --subject_id $SUBJECT_ID`. If you want to include background pixels during the evaluation, add `--include_bkg`.

## Pre-trained checkpoints
* Gyeongsik's avatar [Download](https://drive.google.com/drive/folders/1tLamFJm9-VFXHDcTyCGE8Ar2q0pbkZg3?usp=sharing)
* NeuMan's avatar [Download](https://drive.google.com/drive/folders/1y2c1kYaPV_JRWD1jDDgNceLKwIv6e5gf?usp=sharing)

## Reference
```
@inproceedings{moon2024exavatar,
  title={Expressive Whole-Body 3D Gaussian Avatar},
  author = {Moon, Gyeongsik and Shiratori, Takaaki and Saito, Shunsuke},  
  booktitle={ECCV},
  year={2024}
}

```
