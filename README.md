# Expressive Whole-Body 3D Gaussian Avatar (ECCV 2024)

## [Project Page](https://mks0601.github.io/ExAvatar) | [Paper](https://arxiv.org/abs/2407.21686) | [Video](https://www.youtube.com/watch?v=GzXlAK-sBKY) 


* This is an reimplmentation of **[Expressive Whole-Body 3D Gaussian Avatar](https://mks0601.github.io/ExAvatar/) (ECCV 2024)** by the [first author](https://mks0601.github.io/) after leaving Meta. All the code use public assets without any Meta's internal-only modules. All the results in the camera-ready version and demo videos in the website are from this reimplmented code.
* ExAvatar is designed as a combination of 1) whole-body (body, hands, and face) drivability of [SMPL-X](https://smpl-x.is.tue.mpg.de/) and 2) strong appearance modeling capability of [3DGS](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/).

<p align="middle">
<img src="assets/teaser_compressed.gif" width="960" height="400">
</p>
<p align="center">
Yes, it's me, Gyeongsik in the video :), taken in front of my apartment with my mobile phone.
For more high-resolution demo videos, please visit our <A href="https://mks0601.github.io/ExAvatar">website</A>.
</p>

## Creating and animating avatars from a phone scan
1. To create an avatar, you first need to fit SMPL-X to a video with a single person. Go to [here](./fitting/).
2. Then, go to [here](./avatar) to create and animate the avatar.

## Creating and animating avatars from [X-Humans dataset](https://skype-line.github.io/projects/X-Avatar/)
* Go to [here](https://github.com/mks0601/ExAvatar_RELEASE/tree/X-Humans)

## Reference
```
@inproceedings{moon2024exavatar,
  title={Expressive Whole-Body 3D Gaussian Avatar},
  author = {Moon, Gyeongsik and Shiratori, Takaaki and Saito, Shunsuke},  
  booktitle={ECCV},
  year={2024}
}

```
