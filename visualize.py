import torch
import open3d as o3d


def data_sampler(batch_size, num_points):
    half_batch_size = int(batch_size / 2)
    normal_sampled = torch.randn(half_batch_size, num_points, 3)
    uniform_sampled = torch.rand(half_batch_size, num_points, 3)
    normal_labels = torch.ones(half_batch_size)
    uniform_labels = torch.zeros(half_batch_size)

    input_data = torch.cat((normal_sampled, uniform_sampled), dim=0)
    labels = torch.cat((normal_labels, uniform_labels), dim=0)

    data_shuffle = torch.randperm(batch_size)

    # return input_data[data_shuffle].view(-1, 3), labels[data_shuffle].view(-1, 1)
    return input_data[data_shuffle], labels[data_shuffle]


def create_open3d_point_cloud(points, color):
    """
    Create an Open3D point cloud from a tensor of points.
    Args:
        points (torch.Tensor): Tensor of shape (N, 3).
        color (list): RGB color for the point cloud.
    Returns:
        o3d.geometry.PointCloud: Open3D point cloud object.
    """
    points = points.view(-1, 3)
    points = points.to(torch.float64).cpu().numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


input_data, labels = data_sampler(64, 64)

normal_points = input_data[labels.squeeze() == 1]
uniform_points = input_data[labels.squeeze() == 0]

normal_pcd = create_open3d_point_cloud(normal_points, [1, 0, 0])
uniform_pcd = create_open3d_point_cloud(uniform_points, [0, 0, 1])

o3d.visualization.draw_geometries([normal_pcd, uniform_pcd])
