# Live4D

## Abstract

We present Live4D, a novel framework for text-to-4D (T24D) generation that addresses the challenges of maintaining spatiotemporal coherence, high fidelity, and computational efficiency in dynamic 3D scene generation. Our approach introduces a five-layer residual attention mechanism that explicitly models spatial, temporal, and cross-modal relationships, enabling accurate and efficient 4D content generation. By leveraging Klotski sampling and dynamic matrix decomposition, we achieve high-quality 4D animations while significantly reducing computational overhead.
The primary contributions of this work are threefold: (1) a comprehensive analysis of current 4D generation methods across different input modalities, highlighting their strengths and limitations; (2) a novel five-layer residual attention mechanism that effectively captures viewpoint-time couplings, high-dimensional attention patterns, and multimodal condition fusion; and (3) an innovative Klotski sampling strategy that enables efficient matrix decomposition and localized attention computation, facilitating any-to-any 4D generation.
Extensive experimental results demonstrate the superiority of our approach in terms of spatiotemporal consistency, computational efficiency, and text alignment accuracy. Our framework outperforms state-of-the-art methods in generating complex dynamic scenes while preserving visual consistency across timestamps. This work opens new possibilities for applications in animation, gaming, virtual reality, and other domains requiring immersive, dynamic 3D experiences.

## Method

The T24D training pipeline utilizes a 4D diffusion process to generate temporally and spatially consistent animations. The pipeline begins with two primary datasets: a "Motion Dataset," which contains moving objects, and a "T24D Dataset" consisting of multi-view images of objects (e.g., horses or dinosaurs) paired with corresponding textual descriptions such as "A horse is running." To achieve diverse perspectives, the T24D Dataset is sampled across multiple views (front, side, overhead). The input is then processed by introducing noise, which is iteratively refined through the pipeline from \( Z_{n-1} \) to \( Z_n \), while incorporating view-conditioned information from the T24D samples. During diffusion training, the model uses CLIP (Contrastive Language–Image Pretraining) for text-image alignment and applies several attention mechanisms to enhance spatial and temporal coherence, including frame attention, view attention, Klotski self-attention, and motion-enhancement cross-attention. Spatial and temporal slicing techniques are employed to maintain coherence across frames and views, enabling the model to produce consistent motion dynamics in the generated outputs. This pipeline enables the generation of multi-view animations that align with text descriptions while preserving consistent motion across different perspectives and time steps.

![Method Diagram](./index_files/network250220.png)

## 4D Generation

### Examples

<table>
  <tr>
    <td align="center"><b>robot3</b></td>
    <td align="center"><b>robot1</b></td>
  </tr>
  <tr>
    <td><img src="data/videos/ours/best2/robot3.gif" alt="robot3"></td>
    <td><img src="data/videos/ours/best2/robot1.gif" alt="robot1"></td>
  </tr>
  <tr>
    <td align="center"><b>alien3</b></td>
    <td align="center"><b>alien5</b></td>
  </tr>
  <tr>
    <td><img src="data/videos/ours/best2/alien3.gif" alt="alien3"></td>
    <td><img src="data/videos/ours/best2/alien5.gif" alt="alien5"></td>
  </tr>
  <tr>
    <td align="center"><b>dinosaur4</b></td>
    <td align="center"><b>dinosaur3</b></td>
  </tr>
  <tr>
    <td><img src="data/videos/ours/best2/dinosaur4.gif" alt="dinosaur4"></td>
    <td><img src="data/videos/ours/best2/dinosaur3.gif" alt="dinosaur3"></td>
  </tr>
  <tr>
    <td align="center"><b>panda4</b></td>
    <td align="center"><b>panda1</b></td>
  </tr>
  <tr>
    <td><img src="data/videos/ours/best2/panda4.gif" alt="panda4"></td>
    <td><img src="data/videos/ours/best2/panda1.gif" alt="panda1"></td>
  </tr>
</table>

> **Note:** To view the animations, please ensure the GIF files are placed in the correct paths as specified.

## Comparisons with Live4D

### a volcano erupts

<table>
  <tr>
    <td align="center"><b>Dreamgaussian4D</b></td>
    <td align="center"><b>Ours</b></td>
  </tr>
  <tr>
    <td><img src="data/videos/ours/compare2/it50000-test-middlemotion2.gif" alt="Dreamgaussian4D - Volcano"></td>
    <td><img src="data/videos/ours/compare2/it50000-test-largemotion1.gif" alt="Ours - Volcano"></td>
  </tr>
</table>

### a panda is lifting weights

<table>
  <tr>
    <td align="center"><b>Dreamgaussian4D</b></td>
    <td align="center"><b>Ours</b></td>
  </tr>
  <tr>
    <td><img src="data/videos/ours/compare2/panda-it100000-test.gif" alt="Dreamgaussian4D - Panda"></td>
    <td><img src="data/videos/ours/compare2/panda3.gif" alt="Ours - Panda"></td>
  </tr>
</table>

### a dinosaur is running

<table>
  <tr>
    <td align="center"><b>Dreamgaussian4D</b></td>
    <td align="center"><b>Ours</b></td>
  </tr>
  <tr>
    <td><img src="data/videos/ours/compare2/it20000-test-smallmotion1.gif" alt="Dreamgaussian4D - Dinosaur"></td>
    <td><img src="data/videos/ours/compare2/dinousaurrunning.gif" alt="Ours - Dinosaur"></td>
  </tr>
</table>

## Acknowledgments

Website template from [DreamFusion](https://dreamfusion3d.github.io/) and [MVDream](https://mv-dream.github.io/). We thank the authors for the open-source code.




## Viusal Effects
### Displayments of the Motion4D Models
Just open the file index.html to view the visual effects and introduction. The videos are restored in ./data.



## stage one Setup
### Requirements

```shell
# create virtual environment
conda create -n Live4D python=3.8
conda activate Live4D
# install packages
pip install -r requirements.txt
```

### Weights of Foundation Models
```shell
git lfs install
## You can choose the ModelScopeT2V or ZeroScope, etc., as the foundation model.
## ZeroScope
git clone https://huggingface.co/cerspense/zeroscope_v2_576w ./models/zeroscope_v2_576w/
## ModelScopeT2V
git clone https://huggingface.co/damo-vilab/text-to-video-ms-1.7b ./models/model_scope/
```
### Weights of trained Live4D <a name="download_weights"></a>
```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/ruizhaocv/Live4D_weights ./outputs

# More and better trained Live4D are released at a new repo:
git clone https://huggingface.co/ruizhaocv/Live4D ./outputs
# The usage is slightly different, which will be updated later.
```

## Usage
### Training

#### Train Live4D on multiple videos:
```bash
python MLive4D_train.py --config ./configs/config_multi_videos.yaml
```
#### Train Live4D on a single video:
```bash
python Live4D_train.py --config ./configs/config_single_video.yaml
```

Note:  
- Before running the above command, 
make sure you replace the path to foundational model weights and training data with your own in the config files `config_multi_videos.yaml` or `config_single_video.yaml`.
- Generally, training on multiple 16-frame videos usually takes `300~500` steps, about `9~16` minutes using 1-8 A100 GPUs. Training on a single video takes `50~150` steps, about `1.5~4.5` minutes using one A100 GPU. The required VRAM for training is around `54GB`.
- Reduce `n_sample_frames` if your GPU memory is limited.
- Reduce the learning rate and increase the training steps for better performance.


### Inference
```bash
python Live4D_inference.py --model /path/to/the/foundation/model  --prompt "Your prompt" --checkpoint_folder /path/to/the/trained/Live4D --checkpoint_index 300 --noise_prior 0.
```
Note: 
- Replace `/path/to/the/foundation/model` with your own path to the foundation model, like ZeroScope.
- The value of `checkpoint_index` means the checkpoint saved at which the training step is selected.
- The value of `noise_prior` indicates how much the inversion noise of the reference video affects the generation. 
We recommend setting it to `0` for Live4D trained on multiple videos to achieve the highest diverse generation, while setting it to `0.1~0.5` for Live4D trained on a single video for faster convergence and better alignment with the reference video.


## Inference with pre-trained Live4D
All available weights are at official [Huggingface Repo](https://huggingface.co/ruizhaocv/Live4D_weights).
Run the [download command](#download_weights), the weights will be downloaded to the folder `outputs`, then run the following inference command to generate videos.

### Live4D trained on multiple videos:
```bash
python Live4D_inference.py --model /path/to/the/ZeroScope  --prompt "A person is riding a bicycle past the Eiffel Tower." --checkpoint_folder ./outputs/train/riding_bicycle/ --checkpoint_index 300 --noise_prior 0. --seed 7192280
```
Note:  
- Replace `/path/to/the/ZeroScope` with your own path to the foundation model, i.e. the ZeroScope.
- Change the `prompt` to generate different videos. 
- The `seed` is set to a random value by default. Set it to a specific value will obtain certain results, as provided in the table below.



### Live4D trained on a single video:
16 frames:
```bash
python Live4D_inference.py --model /path/to/the/ZeroScope  --prompt "A tank is running on the moon." --checkpoint_folder ./outputs/train/car_16/ --checkpoint_index 150 --noise_prior 0.5 --seed 8551187
```


24 frames:
```bash
python Live4D_inference.py --model /path/to/the/ZeroScope  --prompt "A truck is running past the Arc de Triomphe." --checkpoint_folder ./outputs/train/car_24/ --checkpoint_index 150 --noise_prior 0.5 --width 576 --height 320 --num-frames 24 --seed 34543
```


## Live4D for Sports <a name="Live4D_for_Sports"></a>

```bash
python Live4Dinference.py --model /path/to/the/ZeroScope  --prompt "A panda is lifting weights in a garden." --checkpoint_folder ./outputs/train/lifting_weights/ --checkpoint_index 300 --noise_prior 0. --seed 9365597
```



More sports, to be continued ...






## Live4D with Customized Appearance <a name="Live4D_with_Customized_Appearance"></a>
### Train
Train the spatial path with reference images.
```bash
python Live4D_train.py --config ./configs/config_multi_images.yaml
```
Then train the temporal path to learn the motions in reference videos.
```bash
python Live4D_train.py --config ./configs/config_multi_videos.yaml
```

### Inference
Inference with spatial path learned from reference images and temporal path learned form reference videos.
```bash
python Live4D_inference_multi.py --model /path/to/the/foundation/model  --prompt "Your prompt" --spatial_path_folder /path/to/the/trained/Live4D/spatial/lora/ --temporal_path_folder /path/to/the/trained/Live4D/temporal/lora/ --noise_prior 0.
```
### Example

Run the following command.
```bash
python Live4D_inference_multi.py --model /path/to/the/ZeroScope  --prompt "A Terracotta Warrior is riding a horse through an ancient battlefield." --spatial_path_folder ./outputs/train/customized_appearance/terracotta_warrior/checkpoint-default/spatial/lora --temporal_path_folder ./outputs/train/riding_horse/checkpoint-default/temporal/lora/ --noise_prior 0. --seed 1455028
```






## Shoutouts

- This code builds on [diffusers](https://github.com/huggingface/diffusers), [Tune-a-video](https://github.com/showlab/Tune-A-Video) and [Text-To-Video-Finetuning](https://github.com/ExponentialML/Text-To-Video-Finetuning). Thanks for open-sourcing!
- Thanks to [camenduru](https://twitter.com/camenduru) for the [colab demo](https://github.com/camenduru/camenduru-colab).
- Thanks to [yhyu13](https://github.com/yhyu13) for the [Huggingface Repo](https://huggingface.co/Yhyu13/yhyu13_LoRA).
- We would like to thank [AK(@_akhaliq)](https://twitter.com/_akhaliq?lang=en) and huggingface team for the help of setting up oneline gradio demo.
- Thanks to [MagicAnimate](https://github.com/magic-research/magic-animate/) for the gradio demo template.
- Thanks to [deepbeepmeep](https://github.com/deepbeepmeep), and [XiaominLi](https://github.com/XiaominLi1997) for improving the code.





## stage two Install
```bash
# python 3.10 cuda 11.8 
conda create -n dg4d python=3.10 -y && conda activate dg4d
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23 --no-deps --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast/
```

To use pretrained LGM:

```bash
# for LGM
mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..
```



## Image-to-4D
##### (Optional) Preprocess input image
```bash
python scripts/process.py data/anya.png
```
##### Step 1: Generate driving videos
```bash
python scripts/gen_vid.py --path data/anya_rgba.png --seed 42 --bg white
```
##### Step 2: static generation
Static generation with [LGM](https://github.com/3DTopia/LGM):
```bash
python lgm/infer.py big --test_path data/anya_rgba.png
```
Optionally, we support static generation with [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian):
```bash
python dg.py --config configs/dg.yaml input=data/anya_rgba.png
```
See `configs/dghd.yaml` for high-quality DreamGaussian training configurations.

##### Step 3: dynamic generation
```bash
# load static 3D from LGM
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png

# (Optional) to load static 3D from DreamGaussian, add `radius=2`
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png radius=2

# (Optional) to turn on viser GUI, add `gui=True`, e.g.:
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png gui=True
```
See `configs/4d_low.yaml` and `configs/4d_demo.yaml` for more memory-friendly and faster optimization configurations.

##### (Optional) Step 4: mesh refinment
```bash
# export mesh after temporal optimization by adding `mesh_format=obj`
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png mesh_format=obj

# mesh refinement
python main2_4d.py --config configs/refine.yaml input=data/anya_rgba.png

# (Optional) to load static 3D from DreamGaussian, add `radius=2`
python main2_4d.py --config configs/refine.yaml input=data/anya_rgba.png radius=2
```

## Video-to-4D
##### Prepare Data
Download [Consistent4D data](https://consistent4d.github.io/) to `data/CONSISTENT4D_DATA`. `python scripts/add_bg_to_gt.py` will add white background to ground-truth novel views.

##### Step 1: static generation
```bash
python lgm/infer.py big --test_path data/CONSISTENT4D_DATA/in-the-wild/blooming_rose/0.png

# (Optional) static 3D generation with DG
python dg.py --config configs/dg.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose/0.png
```

##### Step 2: dynamic generation
```bash
python main_4d.py --config configs/4d_c4d.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose

# (Optional) to load static 3D from DG, add `radius=2`
python main_4d.py --config configs/4d_c4d.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose radius=2
```



## Tips

- Black video after running `gen_vid.py`.
    - Make sure pytorch version is >=2.0 



## Acknowledgement
## Note
This project contains two stage of text to 4D generation. The first stage T2V is modified from motionderector. The second stage is intensively reconstructed from Motion4D. We appreaciate it for their great effort. 
This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [4DGaussians](https://github.com/hustvl/4DGaussians)
* [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
* [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
* [threestudio](https://github.com/threestudio-project/threestudio)
* [nvdiffrast](https://github.com/NVlabs/nvdiffrast)









