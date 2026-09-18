import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


# 设置 matplotlib 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False


img = cv2.imread(r'D:\pycharm\PythonProject\picture\PT.png')
h, w = img.shape[:2]

np.random.seed(None)  # 固定随机种子，保证结果可复现
shake_mats = []  # 保存所有变换矩阵
offset_images = []  # 保存生成模糊图前的每张偏移图像


# 生成3组变换矩阵 + 对应的偏移图像
for i in range(3):
    # 随机生成平移、旋转参数
    dx = np.random.uniform(-5, 5)  # 随机平移
    dy = np.random.uniform(-5, 5)
    angle = np.random.uniform(-1, 1)  # 随机旋转
    rad = np.deg2rad(angle) # 角度转弧度

    # 构造3x3运动变换矩阵
    M = np.array([
        [np.cos(rad), -np.sin(rad), dx],
        [np.sin(rad), np.cos(rad), dy],
        [0, 0, 1]
    ], dtype=np.float32)
    shake_mats.append(M)

    # 应用变换，生成单张偏移图像并保存
    M_affine = M[:2, :]  # 提取2x3仿射矩阵
    trans_img = cv2.warpAffine(img, M_affine, (w, h), borderMode=cv2.BORDER_REFLECT)
    offset_images.append(trans_img)

# ---------------------- 3. 生成运动模糊图像----------------------
blur_img = np.zeros_like(img, dtype=np.float32)
for trans_img in offset_images:
    blur_img += trans_img.astype(np.float32)
blur_img = (blur_img / len(offset_images)).astype(np.uint8)

# ---------------------- 4. 修复逻辑：对每张偏移图像单独逆变换，再叠加平均 ----------------------
restored_single_images = []  # 保存每张偏移图像逆变换后的结果
for idx, M in enumerate(shake_mats):
    # 取出对应位置的偏移图像
    single_offset_img = offset_images[idx].astype(np.float32)

    # 计算当前变换矩阵的逆矩阵
    M_inv = np.linalg.inv(M)
    M_inv_affine = M_inv[:2, :].astype(np.float32)

    # 对单张偏移图像做逆变换（抵消平移+旋转）
    restored_single = cv2.warpAffine(
        single_offset_img,
        M_inv_affine,
        (w, h),
        borderMode=cv2.BORDER_REFLECT
    )
    restored_single_images.append(restored_single)

# 将所有逆变换后的单张图像叠加平均，得到最终修复图
restored_img = np.zeros_like(img, dtype=np.float32)
for restored_single in restored_single_images:
    restored_img += restored_single
restored_img = (restored_img / len(restored_single_images)).astype(np.uint8)

# 计算PSNR

mse = np.mean((img.astype(np.float32) - restored_img.astype(np.float32)) ** 2)
max_pixel = 255
psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
print(f"\nPSNR值：PSNR = {psnr:.2f}dB")
   


# ---------------------- 5. 显示三张图（原始、模糊、修复后） ----------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# 第一张：原始图像
axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0].set_title("原始图像", fontsize=12)
axes[0].axis('off')  # 关闭坐标轴

# 第二张：运动模糊图像
axes[1].imshow(cv2.cvtColor(blur_img, cv2.COLOR_BGR2RGB))
axes[1].set_title("Simulated Motion-Blurred Image", fontsize=12)
axes[1].axis('off')

# 第三张：修复后的清晰图像
axes[2].imshow(cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB))
axes[2].set_title("Restored Image", fontsize=12)
axes[2].axis('off')

# 调整子图间距，避免标题/图像重叠
plt.tight_layout()
# 显示整个窗口（包含三张图）
plt.show()

#cv2.imwrite("D:/pycharm/PythonProject/clear.png", img)
#cv2.imwrite("D:/pycharm/PythonProject/blurrt.png", blur_img)
