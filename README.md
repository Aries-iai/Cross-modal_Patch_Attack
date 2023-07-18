# Unified Adversarial Patch for  Cross-modal Attacks in the Physical World
## <div align="center">Quick Start Examples</div>

<details open>
<summary>Install</summary>
  
[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/Aries-iai/Cross-modal_Patch_Attack/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
  
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->


```bash
$ git clone https://github.com/Aries-iai/Cross-modal_Patch_Attack
$ cd Cross-modal_Patch_Attack
$ pip install -r requirements.txt
```

<details open> 
<summary>Data Convention</summary>
The data is organized as follows:

```
dataset  
|-- attack_infrared
    |-- 000.png        # images in the infrared modality
    |-- 001.png
    ...
|-- attack_visible
    |-- 000.png        # images in the visible modality
    |-- 001.png
    ...
```

Here, we should ensure the consistency of infrared images and visible images' names.

<details open> 
<summary>Running</summary>

```shell
python spline_DE_attack.py
```

