import os

# 指定文件夹路径
folder_path = 'D:\\User\\下载\\JUniward\\validation\\stego'

# 获取文件夹下所有文件的列表并按文件名排序
files = os.listdir(folder_path)
files.sort()

# 重命名文件
for i, file_name in enumerate(files):
    file_ext = os.path.splitext(file_name)[1]
    new_file_name = str(i + 1).zfill(5) + file_ext
    os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))

print("文件重命名完成。")
