from scipy.spatial.distance import mahalanobis
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import torch
from tqdm import tqdm










#TODO: Doesn't scale over many images, calculate all at once instead of loop
def calculateScoreMaps(mean, cov_inv, embedding_vectors, gaussian_filter=True):
    B, D, H, W = embedding_vectors.shape

    score_maps = torch.zeros(B, H, W)    
    
    for i in range(embedding_vectors.shape[0]):
        mahalonobis_distances = mahalonobis(mean, cov_inv, embedding_vectors[i])
        score_maps[i] = mahalonobis_distances
        
    #TODO: pytorch implementation of gaussian filter
    if gaussian_filter:
        score_maps = score_maps.numpy()    
        for i in range(score_maps.shape[0]):
            score_maps[i] = gaussian_filter(score_maps[i], sigma=4)
        score_maps = torch.from_numpy(score_maps)
        
    return score_maps
    



def mahalonobis(mean, cov_inv, embedding_vectors):
    D, H, W = embedding_vectors.shape
    embedding_vectors = embedding_vectors.reshape(D, H*W).permute(1, 0)    
    mean = mean.permute(1, 0)
    cov_inv = cov_inv.permute(2, 0, 1)    
    
    diff = mean-embedding_vectors
    mult1 = torch.bmm(diff.unsqueeze(1), cov_inv)
    mult2 = torch.bmm(mult1, diff.unsqueeze(2))
    sqrt = torch.sqrt(mult2)
    mahalonobis_distance = sqrt.reshape(H, W)
    
    return mahalonobis_distance

    




# def calculateScore(train_outputs, embedding_vectors):

#     B, C, H, W = embedding_vectors.size()
#     embedding_vectors = embedding_vectors.view(B, C, H * W).numpy()
#     dist_list = []
#     for i in tqdm(range(H*W), 'Calculating distances'):
#         mean = train_outputs[0][:, i]
#         conv_inv = np.linalg.inv(train_outputs[1][:, :, i])
#         dist = [mahalanobis(sample[:, i], mean, conv_inv) for sample in embedding_vectors]
#         dist_list.append(dist)

#     dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)

#     # upsample
#     dist_list = torch.tensor(dist_list)
# #     print(embedding_vectors.shape)
#     score_map = F.interpolate(dist_list.unsqueeze(1), size=H, mode='bilinear',
#                               align_corners=False).squeeze().numpy()#TODO: Check size

#     # apply gaussian smoothing on the score map
#     for i in range(score_map.shape[0]):
#         score_map[i] = gaussian_filter(score_map[i], sigma=4)

#     # Normalization
# #     max_score = score_map.max()
# #     min_score = score_map.min()
# #     scores = (score_map - min_score) / (max_score - min_score)
#     return score_map