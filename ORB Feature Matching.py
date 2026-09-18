
import numpy as np
import cv2
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False


def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_rgb, gray


def estimate_shake_matrix(gray_clear, gray_blur):

    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, patchSize=31)

    kp_clear, des_clear = orb.detectAndCompute(gray_clear, None)
    kp_blur, des_blur = orb.detectAndCompute(gray_blur, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(des_clear, des_blur)

    matches = sorted(matches, key=lambda x: x.distance)[:int(len(matches)*0.8)]

    pts_clear = np.float32([kp_clear[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_blur = np.float32([kp_blur[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.estimateAffine2D(pts_blur, pts_clear, cv2.RANSAC, ransacReprojThreshold=5.0)

    return M, matches, kp_clear, kp_blur, mask

def correct_shake_image(img_blur, M, clear_img_shape):
    corrected_img = cv2.warpAffine(img_blur, M, (clear_img_shape[1], clear_img_shape[0]))
    return corrected_img


def calculate_psnr(img_clear, img_corrected):
    mse = np.mean((img_clear.astype(np.float32) - img_corrected.astype(np.float32)) ** 2)
    max_pixel = 255
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def visualize_results(img_clear, img_blur, img_corrected, matches, kp_clear, kp_blur, mask, M, psnr):
    mask_list = mask.ravel().tolist()
    match_img = cv2.drawMatches(
        img_clear, kp_clear, img_blur, kp_blur, matches, None,
        matchColor=(0, 255, 0), singlePointColor=None, matchesMask=mask_list, flags=2
    )

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(match_img)
    plt.title("ORB Feature Matching Results")
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.imshow(img_clear)
    plt.title("Camera Shake–Blurred Image（1.jpg）")
    plt.axis('off')
    plt.subplot(2, 2, 3)
    plt.imshow(img_blur)
    plt.title("原始清晰图像（2.jpg）")
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.title("Restored Image")
    plt.imshow(img_corrected)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    print("手抖矩阵：")
    print(M)
    #print(f"\nPSNR值：PSNR = {psnr:.2f}dB")

if __name__ == "__main__":
    clear_image_path = r"D:\pycharm\PythonProject\blurr.png" 
    blur_image_path = r"D:\pycharm\PythonProject\clear.png"  
    img_clear, gray_clear = load_and_preprocess(clear_image_path)
    img_blur, gray_blur = load_and_preprocess(blur_image_path)
    shake_matrix, matches, kp_clear, kp_blur, mask = estimate_shake_matrix(gray_clear, gray_blur)
    img_corrected = correct_shake_image(img_blur, shake_matrix, img_clear.shape)
    psnr_value = calculate_psnr(img_clear, img_corrected)
    img_corrected = r"D:\pycharm\PythonProject\blurrt.png"
    img_corrected, gray_corrected = load_and_preprocess(img_corrected)
    visualize_results(img_clear, img_blur, img_corrected, matches, kp_clear, kp_blur, mask, shake_matrix, psnr_value)
