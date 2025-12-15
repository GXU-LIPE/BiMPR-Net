# encoding=gbk
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import torchvision.utils as vutils


# def visualize_image(image, output_path, index_i):
#     # for j in range(image.shape[0]):
#     for j, image_j in enumerate(image):
#         index_i = index_i + 1
#         if 1 <= index_i <= 2500:
#             image_path = os.path.join(output_path, f"image_{index_i}.png")
#             vutils.save_image(image[j], image_path, normalize=False)


def visualize_image(source_path, target_path, index_i, ran_key):
    for j, image_j in enumerate(source_path):
        index_i = index_i + 1
        if 8000 <= index_i <= 9000:
            image_path = os.path.join(target_path, f"{index_i}.png")
            shutil.copyfile(image_j, image_path)


def visualize_point_cloud(complete, output_path, index_p, ran_key):
    for j, complete_j in enumerate(complete):
        index_p = index_p + 1
        if 0 <= index_p <= 1000:
            fig = plt.figure(figsize=(8, 8))
            complete_j = complete_j.squeeze().cpu().numpy()
            x, z, y = complete_j.transpose(1, 0)
            ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
            ax.axis('off')
            # ax.axis('scaled')
            ax.view_init(30, 45)
            max, min = np.max(complete_j), np.min(complete_j)
            ax.set_xbound(min, max)
            ax.set_ybound(min, max)
            ax.set_zbound(min, max)
            ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')
            # if 1 <= index_p <= 100:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')
            # elif 100 <= index_p <= 200:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='viridis')
            # elif 200 <= index_p <= 300:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='plasma')
            # elif 300 <= index_p <= 400:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='inferno')
            # elif 400 <= index_p <= 500:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='magma')
            # elif 500 <= index_p <= 600:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='cividis')  ###
            # elif 600 <= index_p <= 700:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='coolwarm')
            # elif 700 <= index_p <= 800:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='RdYlBu')  #####
            # elif 800 <= index_p <= 900:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='gist_rainbow')
            # elif 900 <= index_p <= 1000:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='PuOr')  #####
            # elif 1000 <= index_p <= 1100:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='Reds')
            # elif 1100 <= index_p <= 1200:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='Blues')
            # elif 1200 <= index_p <= 1300:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='Greens')
            # elif 1300 <= index_p <= 1400:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='Oranges')
            # elif 1400 <= index_p <= 1500:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='Greys')
            # elif 1500 <= index_p <= 1600:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='gist_earth')
            # elif 1600 <= index_p <= 1700:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='terrain')  ###
            # elif 1700 <= index_p <= 1800:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='ocean')  ###
            # elif 1800 <= index_p <= 1900:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='gnuplot')  ###
            # elif 1900 <= index_p <= 2000:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='gnuplot2')
            # elif 2000 <= index_p <= 2100:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='nipy_spectral')
            # elif 2100 <= index_p <= 2200:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='flag')
            # elif 2200 <= index_p <= 2300:
            #     ax.scatter(x, y, z, zdir='z', c=x, cmap='brg')

            fig.canvas.draw()
            img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img = Image.fromarray(img)
            img.save(os.path.join(output_path, f'{index_p}.png'))
            plt.close(fig)


def save_point_cloud_as_obj(point_cloud_data, save_path, index_o, file_prefix="point_cloud"):
    """
    将点云数据保存为多个OBJ文件

    参数：
    point_cloud_data : numpy.ndarray
        包含点云数据的数组，形状为[B, N, C]，其中B表示点云数量，N表示点的数量，C表示坐标维度。
    file_prefix : str, optional
        OBJ文件名的前缀，默认为"point_cloud"。

    返回：
    无返回值，将点云数据保存为多个OBJ文件。
    """
    for i, cloud in enumerate(point_cloud_data):
        index_o = index_o + 1
        if 1 <= index_o <= 100:
            obj_filename = os.path.join(save_path, f"{file_prefix}_{index_o}.obj")

            with open(obj_filename, "w") as file:
                for point in cloud:
                    file.write(f"v {point[0]} {point[1]} {point[2]}\n")


def save_point_cloud_as_ply(point_cloud_data, save_path, index_ply, ran_key):
    """
    将点云数据保存为多个PLY文件

    参数：
    point_cloud_data : numpy.ndarray
        包含点云数据的数组，形状为[B, N, C]，其中B表示点云数量，N表示点的数量，C表示坐标维度。
    save_path : str
        指定保存PLY文件的路径。
    file_prefix : str, optional
        PLY文件名的前缀，默认为"point_cloud"。

    返回：
    无返回值，将点云数据保存为多个PLY文件。
    """

    for i, cloud in enumerate(point_cloud_data):
        index_ply = index_ply + 1
        if 0 <= index_ply <= 1000:
        # if index_ply % 50 == 0:
            ply_filename = os.path.join(save_path, f"{index_ply}.ply")

            with open(ply_filename, "w") as file:
                file.write("ply\n")
                file.write("format ascii 1.0\n")
                file.write("element vertex %d\n" % len(cloud))
                file.write("property float x\n")
                file.write("property float y\n")
                file.write("property float z\n")
                file.write("end_header\n")

                for point in cloud:
                    file.write(f"{point[0]} {point[1]} {point[2]}\n")


# import open3d as o3d
# import numpy as np
#
# def visualize_point_cloud(complete, output_path, index_p):
#     for j, complete_j in enumerate(complete):
#         index_p = index_p + 1
#         if index_p % 500 == 0:
#             complete_j = complete_j.squeeze().cpu().numpy()
#             pcd = o3d.geometry.PointCloud()
#             pcd.points = o3d.utility.Vector3dVector(complete_j)
#
#             # Set visualization parameters
#             visualizer = o3d.visualization.Visualizer()
#             visualizer.create_window()
#             render_option = visualizer.get_render_option()
#             render_option.point_size = 2.0
#
#             # Add point cloud to visualization
#             visualizer.add_geometry(pcd)
#
#             # Take screenshot and save to file
#             visualizer.poll_events()
#             visualizer.update_renderer()
#             image = visualizer.capture_screen_float_buffer()
#             o3d.io.write_image(os.path.join(output_path, f'point_cloud_{index_p}.png'), image)
#
#             visualizer.destroy_window()


# import open3d as o3d
#
# def visualize_point_cloud(complete, output_path, index_p):
#     for j, complete_j in enumerate(complete):
#         index_p = index_p + 1
#         if index_p % 500 == 0:
#             point_cloud = o3d.geometry.PointCloud()
#             complete_j = complete_j.squeeze().cpu().numpy()
#             point_cloud.points = o3d.utility.Vector3dVector(complete_j)
#             colors = complete_j[:, 0]  # Use the x-axis values for color mapping
#             colors = (colors - np.min(colors)) / (np.max(colors) - np.min(colors))  # Normalize color values
#             colors = plt.cm.viridis(colors)[:, :3]  # Convert colormap values to RGB format
#             point_cloud.colors = o3d.utility.Vector3dVector(colors)
#             o3d.visualization.draw_geometries([point_cloud], width=800, height=800, front=[-0.3, -0.3, -1.0], lookat=[0, 0, 0])
#             o3d.io.write_image(os.path.join(output_path, f'point_cloud_{index_p}.png'), o3d.visualization.RenderOption(
#                 {"point_show_normal": False, "background_color": np.asarray([255, 255, 255]), "show_coordinate_frame": False}))


# import numpy as np
# import os
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
# from PIL import Image
#
#
# def visualize_point_cloud(complete, output_path, index_p):
#     # 设置超采样比率
#     dpi = 300
#     # 设置图像大小
#     figsize = (6, 6)
#
#     for j, complete_j in enumerate(complete):
#         index_p = index_p + 1
#         if index_p % 500 == 0:
#             fig = plt.figure(figsize=figsize, dpi=dpi)
#             complete_j = complete_j.squeeze().cpu().numpy()
#             x, z, y = complete_j.transpose(1, 0)
#             ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
#             ax.axis('off')
#             ax.view_init(30, 45)
#             max, min = np.max(complete_j), np.min(complete_j)
#             ax.set_xbound(min, max)
#             ax.set_ybound(min, max)
#             ax.set_zbound(min, max)
#             ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')
#             # 渲染图像
#             canvas = FigureCanvas(fig)
#             canvas.draw()
#             # 获得渲染结果
#             s, (width, height) = canvas.print_to_buffer()
#             # 将渲染结果转换为numpy数组
#             img = np.fromstring(s, np.uint8).reshape((height, width, 4))
#             # 从numpy数组创建PIL图像
#             img = Image.fromarray(img).convert('RGB')
#             # 对图像进行大小调整
#             img = img.resize((figsize[0] * dpi, figsize[1] * dpi), resample=Image.LANCZOS)
#             # 保存图像
#             img.save(os.path.join(output_path, f'point_cloud_{index_p}.png'))
#             plt.close(fig)


# from PIL import Image, ImageFilter
#
# def visualize_point_cloud(complete, output_path, index_p):
#     for j, complete_j in enumerate(complete):
#         index_p = index_p + 1
#         if index_p % 500 == 0:
#             fig = plt.figure(figsize=(8, 8))
#             complete_j = complete_j.squeeze().cpu().numpy()
#             x, z, y = complete_j.transpose(1, 0)
#             ax = fig.add_subplot(projection=Axes3D.name, adjustable='box')
#             ax.axis('off')
#             # ax.axis('scaled')
#             ax.view_init(30, 45)
#             max, min = np.max(complete_j), np.min(complete_j)
#             ax.set_xbound(min, max)
#             ax.set_ybound(min, max)
#             ax.set_zbound(min, max)
#             ax.scatter(x, y, z, zdir='z', c=x, cmap='jet')
#
#             fig.canvas.draw()
#             img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
#             img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#             img = Image.fromarray(img)
#
#             # Apply Gaussian filter
#             img = img.filter(ImageFilter.GaussianBlur(radius=2))
#
#             # Apply sharpen filter
#             img = img.filter(ImageFilter.SHARPEN)
#
#             img.save(os.path.join(output_path, f'point_cloud_{index_p}.png'), dpi=(300,300))
#             plt.close(fig)
