import torch.nn.functional as F
import torch
from torchvision.ops import roi_align, roi_pool


def my_roi_pool(feature_map, target_size, pool_op="roi_pool"):
    N, C, H, W = feature_map.shape
    box = torch.tensor([[i, 0, 0, H, W] for i in range(N)],
                       dtype=torch.float, device='cuda')
    target_size = torch.tensor(target_size, device='cuda')

    pool_ops = {"roi_pool": roi_pool, "roi_align": roi_align}

    assert pool_op in pool_ops, f"invalid pool_op: {pool_op}"

    pooled_feature = pool_ops[pool_op](feature_map, box, target_size)
    return pooled_feature


def my_roi_pool_2(feature_map, target_size, pool_op="roi_pool"):
    return F.adaptive_max_pool2d(feature_map, target_size)


if __name__ == "__main__":
    feature = torch.rand((1, 1, 10, 10)).cuda()
    roi_features = my_roi_pool(feature, (5, 5), pool_op='roi_pool')
    roi_features_2 = my_roi_pool_2(feature, (5, 5))
    print(roi_features.shape, roi_features_2.shape)
    print(feature)
    print(roi_features)
    print(roi_features_2)
    # print(torch.testing.assert_allclose(roi_features, roi_features_2))
    # print(my_roi_pool(feature, (32, 32), pool_op='roi_pool').shape)
    # print(feature[0])
    # print(roi_features[0])
    #
    # print(feature[1])
    # print(roi_features[1])
