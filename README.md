# Creating an avatar from [X-Humans dataset](https://skype-line.github.io/projects/X-Avatar/)

* This branch includes **avatar creation pipeline and animation function**.
* This code includes **avatar creation pipeline and animation function** only when exact foreground mask is given like [X-Humans dataset](https://skype-line.github.io/projects/X-Avatar/).


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
|-- |-- XHumans
|-- |-- |-- data/00028
|-- |-- |-- data/00034
|-- |-- |-- data/00087
|-- tools
|-- output
```
* `main` contains high-level code for the avatar creation/animation and configurations.
* `common` contains kernel code. Download SMPL-X 1.1 version from [here](https://smpl-x.is.tue.mpg.de/download.php). Download FLAME 2020 version from [here](https://flame.is.tue.mpg.de/download.php).
* `data` contains data loading code.
* `tools` contains pre-processing and evaluation code.
* `output` contains log, visualized outputs, and fitting result.

## XHumans videos
* You can download original XHumans data from [here](https://skype-line.github.io/projects/X-Avatar/).
* First, use the official pre-processed code (`_X_Humans_preprocess_XHumans.py`). As the original code has a bug, which does not clamp UV coordinates in [0,1], we added a single line to handle this. Please use our `tools/_X_Humans_preprocess_XHumans.py` instead of the original one.
* We provide our additional pre-processed XHumans data in [here](https://drive.google.com/drive/folders/1TalHPkbohPoTPNawVi2gbj6M8nAyYAE9?usp=sharing).
* We used the same pre-processing stage of the [fitting](https://github.com/mks0601/ExAvatar_RELEASE/tree/main/fitting).

## Train
* Set `dataset='XHumans'` in `main/config.py`.
* Go to `main` folder and run `python train.py --subject_id $SUBJECT_ID`. The checkpoints are saved in `output/model/$SUBJECT_ID`.
* You can use one of `00028`, `00034`, and `00087` for `$SUBJECT_ID`.

## Visualize a rotating avatar with the neutral pose
* Set `dataset='XHumans'` in `main/config.py`.
* Go to `main` folder and run `python get_neutral_pose.py --subject_id $SUBJECT_ID --test_epoch 24`.
* You can see a rotating avatar with the neutral pose in `./main/neutral_pose`.

## Animation
* Set `dataset='XHumans'` in `main/config.py`.
* Go to `main` folder and run `python animation.py --subject_id $SUBJECT_ID --test_epoch 24 --motion_path $PATH` if you want to use an avatar in `output/model_dump/$SUBJECT_ID`. `$PATH` should contain SMPL-X parameters to animate the avatar. You can prepare `$PATH` with [here](../fitting).
* To render the avatar from rotating camera, run `python animate_view_rot.py --subject_id $SUBJECT_ID --test_epoch 24 --motion_path $PATH`.
* We provide SMPL-X parameters of several videos (examples of `$PATH`) in [here](https://drive.google.com/drive/folders/1ApDtoyqrcP2r2ZvX24eptmSefJvw_no5?usp=sharing).

## Test and evaluation
* Set `dataset='XHumans'` in `main/config.py`.
* You can see test results on the testing frames by running `python test.py --subject_id $SUBJECT_ID --test_epoch 24`. The results are saved to `output/result/$SUBJECT_ID`.
* For the evaluation of the X-Humans dataset, go to `tools` folder and run `python eval_xhumans.py --output_path ../output/result/$SUBJECT_ID --subject_id $SUBJECT_ID`.

## Pre-trained checkpoints
* X-Humans' avatar [Download](https://drive.google.com/drive/folders/1o_u3TIzONc21-GAgQ5ukVq2SZXdEVfFT?usp=sharing)

## Reference
```
@inproceedings{moon2024exavatar,
  title={Expressive Whole-Body 3D Gaussian Avatar},
  author = {Moon, Gyeongsik and Shiratori, Takaaki and Saito, Shunsuke},  
  booktitle={ECCV},
  year={2024}
}

```

