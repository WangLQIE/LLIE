# A U-Net-Based Low-Light Image Enhancement Method: Integrating Transformer and Frequency-Domain Attention Mechanisms

Xuejian Wang

<hr />

> **Abstract:** *To address the issues of noise amplification, color distortion, and detail loss commonly encountered in low-light image enhancement, this paper proposes an improved U-Net model that integrates Transformer architecture and frequency-domain attention mechanisms. The method is built upon the U-Net framework, incorporating ResLeWin Transformer modules in both the encoder and decoder to achieve stronger local feature representation. A frequency-domain channel attention module is further introduced to weight low-frequency components, thereby improving the reconstruction quality of the overall image structure. In the stages with lower feature-map resolution, a global filtering module is employed to suppress high-frequency noise and enhance texture consistency through frequency-domain multiplicative operations. To improve training performance, a hybrid loss function combining spatial- and frequency-domain feature differences is designed, including pixel-wise error, structural similarity, and frequency-domain mean error. Experiments conducted on two real-world low-light image datasets, SID-Sony and SID-Fuji, demonstrate that the proposed method achieves significant improvements over several state-of-the-art approaches in terms of PSNR and SSIM, while also delivering superior visual performance in brightness, color restoration, and detail preservation.* 
<hr />

## Datasets
We provide the divided dataset for SID.
   
 SID  (  https://github.com/cchen156/Learning-to-See-in-the-Dark?tab=readme-ov-file  ) 

## Pretrained Models
We provide  pretrained models for SID-Sony and SID-Fuji.
   
| Model | Sony | Fuji | 

| U_GFR | [Sony](  https://pan.baidu.com/s/1tfIEiQNjwR_pxstdKF4rDw?pwd=123S   ) | [Fuji](  https://pan.baidu.com/s/1L9VEzVpRe76HuGJ0_wuRfA?pwd=123F   ) |

提取码:123S,123F

## Training and Testing

After preparing the training data or testing data, use 
```
CUDA_VISIBLE_DEVICES=1  python LLIE_GFR/main.py
```

## Contact
Should you have any questions, please contact 1207579310@qq.com
 
