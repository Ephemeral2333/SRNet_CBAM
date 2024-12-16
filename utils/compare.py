import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 读取图像
image1 = mpimg.imread('/root/autodl-tmp/wow4_dataset/train/cover/1.pgm')
image2 = mpimg.imread('/root/autodl-tmp/wow4_dataset/train/stego/1.pgm')

# 确保两个图像具有相同的维度
assert image1.shape == image2.shape, "Images must have the same dimensions."

# 计算残差图像
# 这里使用np.abs计算绝对值差异，以确保所有差异都是正值
residual_image = np.abs(image1 - image2)

# 显示图像1
plt.imshow(image1)
plt.gray()
plt.title('Cover Image')
plt.show()

# 显示图像2
plt.imshow(image2)
plt.gray()
plt.title('Stego Image')
plt.show()


# 显示残差图像
plt.imshow(residual_image)
# plt.imshow(re)
plt.title('Residual Image')
# plt.axis('off') # 不显示坐标轴
plt.show()
