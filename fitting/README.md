# Fitting SMPL-X to a monocular video

* This branch provides code to fit SMPL-X parametric model to a monocular video.
* The fitting should be done first before creating the avatar.

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
|-- |-- |-- flame/2019/generic_model.pkl
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
|-- |-- XHumans
|-- |-- |-- data/00028
|-- |-- |-- data/00034
|-- |-- |-- data/00087
|-- tools
|-- |-- code_to_copy
|-- |-- DECA
|-- |-- Hand4Whole_RELEASE
|-- |-- mmpose
|-- |-- segment-anything
|-- |-- Depth-Anything-V2
|-- output
```
* `main` contains high-level code for the fitting and configurations.
* `common` contains kernel code. Download SMPL-X 1.1 version from [here](https://smpl-x.is.tue.mpg.de/download.php). Download FLAME 2019 and 2020 version from [here](https://flame.is.tue.mpg.de/download.php).
* `data` contains data loading code.
* `tools` contains pre-processing code.
* `output` contains log, visualized outputs, and fitting result.

## Custom videos (your own videos)
* Go to `tools` folder and clone [DECA](https://github.com/yfeng95/DECA), [Hand4Whole](https://github.com/mks0601/Hand4Whole_RELEASE), [mmpose](https://github.com/open-mmlab/mmpose), [segment-anything](https://github.com/facebookresearch/segment-anything), and [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2). Install them. For the `mmpose`, we use `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` and `dw-ll_ucoco_384.pth`. For the `segment-anything`, we use `sam_vit_h_4b8939.pth`. For the `Depth-Anything-V2`, we use `depth_anything_v2_vitl.pth`. Please intsall [COLMAP](https://colmap.github.io/cli.html) with `sudo apt-get install colmap`.
* Run `python copy_code.py` to copy customized code in `code_to_copy` to the above repos.
* Place your video at `data/Custom/data/$SUBJECT_ID/video.mp4`.
* Go to `tools` folder and run `python extract_frames.py --root_path $ROOT/data/Custom/data/$SUBJECT_ID` to extract frames.
* Go to `data/Custom/data/$SUBJECT_ID` and make `frame_list_all.txt`, which includes frame indices to use for the fitting. We recommend to use frames where most of human is visible without being truncated. Each line should contain a frame index and a new line, for example '1\n'.
* Likewise, prepare `frame_list_train.txt` and `frame_list_test.txt` in the same way as `frame_list_all.txt`. Each will be used for training and evaluating the avatar, respectively. We recommend making them in 5 fps (assuming the original video and `frame_list_all.txt` are in 30 fps) to make the avatar creation faster.
* Set `dataset = 'Custom'` in `main/config.py`.
* (Camera option 1) If you want to make this video to create an avatar and the camera is moving, go to `tools` foler and run `python run.py --root_path $ROOT/data/Custom/data/$SUBJECT_ID --use_colmap`. In this way, we get camera parameters from COLMAP.
* (Camera option 2) If 1) you want to make this video to create an avatar and the camera is static or 2) you want to get smplx parameters to animate your avatar (doesn't matter the camera is moving), go to `tools` folder and run `python run.py --root_path $ROOT/data/Custom/data/$SUBJECT_ID`. In this way, we get virtual camera parameters.
* `run.py` will output camera parameters, initial FLAME parameters, initial SMPL-X parameters, and 2D whole-body keypoints, optimized/smoothed SMPL-X parameters, and unwrapped face texture at `data/Custom/data/$SUBJECT_ID`.
* We provide an example of pre-processed custom video in [here](https://drive.google.com/file/d/1YGJZWWpw_R63HiZu65smV6Lrksqa6EOu/view?usp=sharing).

## NeuMan videos
* You can download original NeuMan data from [here](https://github.com/apple/ml-neuman).
* We provide pre-processed Neuman data in [here](https://drive.google.com/drive/folders/15-V9EG21hT4pVhuBdHY3-lpvKjCuHbEU?usp=sharing).
* We provide train/test/validation split files in [here](https://drive.google.com/drive/folders/1L5KC4QIRX_ljQ_vyrIXV11FgynnuCDb8?usp=sharing), made following [the official code](https://github.com/apple/ml-neuman/blob/0149d258b2afe6ef65c91557bba9f874675871e4/data_io/neuman_helper.py#L149).
* We used the same pre-processing stage of the above one for the custom videos after setting `dataset = 'NeuMan'` in `main/config.py`.

## XHumans videos
* You can download original XHumans data from [here](https://skype-line.github.io/projects/X-Avatar/).
* We provide pre-processed XHumans data in [here](https://drive.google.com/drive/folders/1TalHPkbohPoTPNawVi2gbj6M8nAyYAE9?usp=sharing).
* We used the same pre-processing stage of the above one for the custom videos after setting `dataset = 'XHumans'` in `main/config.py`.

## Troubleshoots
Q. Face of fitted smplx looks bad [[issue13](https://github.com/mks0601/ExAvatar_RELEASE/issues/13)]

A. Checkout renderings in `data/Custom/data/flame_init/renders`. Some people run the fitting without preparing DECA's results.


Q. RuntimeError: indices should be either on cpu or on the same device as the indexted tensor

A. Run `export CUDA_VISIBLE_DEVICES=0` before running the fitting.

## Reference
```
@inproceedings{moon2024exavatar,
  title={Expressive Whole-Body 3D Gaussian Avatar},
  author = {Moon, Gyeongsik and Shiratori, Takaaki and Saito, Shunsuke},  
  booktitle={ECCV},
  year={2024}
}

```
