import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim



plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


img = cv2.imread(r'D:\pycharm\PythonProject\picture\PT.png')
h, w = img.shape[:2]

np.random.seed(None) 
shake_mats = []  
offset_images = []  



for i in range(3):

    dx = np.random.uniform(-5, 5)  
    dy = np.random.uniform(-5, 5)
    angle = np.random.uniform(-1, 1) 
    rad = np.deg2rad(angle) 


    M = np.array([
        [np.cos(rad), -np.sin(rad), dx],
        [np.sin(rad), np.cos(rad), dy],
        [0, 0, 1]
    ], dtype=np.float32)
    shake_mats.append(M)


    M_affine = M[:2, :] 
    trans_img = cv2.warpAffine(img, M_affine, (w, h), borderMode=cv2.BORDER_REFLECT)
    offset_images.append(trans_img)


blur_img = np.zeros_like(img, dtype=np.float32)
for trans_img in offset_images:
    blur_img += trans_img.astype(np.float32)
blur_img = (blur_img / len(offset_images)).astype(np.uint8)


restored_single_images = [] 
for idx, M in enumerate(shake_mats):

    single_offset_img = offset_images[idx].astype(np.float32)


    M_inv = np.linalg.inv(M)
    M_inv_affine = M_inv[:2, :].astype(np.float32)


    restored_single = cv2.warpAffine(
        single_offset_img,
        M_inv_affine,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )
    restored_single_images.append(restored_single)

restored_img = np.zeros_like(img, dtype=np.float32)
for restored_single in restored_single_images:
    restored_img += restored_single
restored_img = (restored_img / len(restored_single_images)).astype(np.uint8)



mse = np.mean((img.astype(np.float32) - restored_img.astype(np.float32)) ** 2)
max_pixel = 255
psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
print(f"\nPSNR值：PSNR = {psnr:.2f}dB")
   



fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("原始图像", fontsize=12)
axes[0].axis('off') 


axes[1].imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
axes[1].set_title("Simulated Motion-Blurred Image", fontsize=12)
axes[1].axis('off')


axes[2].imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
axes[2].set_title("Restored Image", fontsize=12)
axes[2].axis('off')


plt.tight_layout()

plt.show()

#cv2.imwrite("D:/pycharm/PythonProject/clear.png", img)
#cv2.imwrite("D:/pycharm/PythonProject/blurrt.png", blur_img)
