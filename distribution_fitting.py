import torch
import numpy as np
from tqdm import tqdm



def getMeanCov(embedding_vectors, device):
    B, C, H, W = embedding_vectors.size()
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    print('Calculating mean')
    mean = torch.mean(embedding_vectors, dim=0)#.numpy()
    cov = torch.zeros(C, C, H * W).numpy()
    I = np.identity(C)

    for i in tqdm(range(H*W), 'Calculating covariance'):
        # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
        cov[:, :, i] = np.cov(embedding_vectors[:, :, i].cpu().numpy(), rowvar=False) + 0.01 * I
        
#     print('Calculating inverse of covariance')
    cov = torch.from_numpy(cov).to(device)
    cov_inv = cov.permute(2,0,1)
    cov_inv = torch.inverse(cov_inv)
    cov_inv = cov_inv.permute(1,2,0)
    
#     cov_inv = torch.inverse(cov_inv.permute(2,0,1))
#     cov_inv = cov_inv.permute(1,2,0)
    mean = mean.to(device)

    return mean, cov, cov_inv


    
    
    
# def getMeanCov(embedding_vectors):
#     B, C, H, W = embedding_vectors.size()
#     embedding_vectors = embedding_vectors.view(B, C, H * W)
#     print('Calculating mean')
#     mean = torch.mean(embedding_vectors, dim=0).numpy()
#     cov = torch.zeros(C, C, H * W).numpy()
#     I = np.identity(C)

#     for i in tqdm(range(H*W), 'Calculating covariance'):
#         # cov[:, :, i] = LedoitWolf().fit(embedding_vectors[:, :, i].numpy()).covariance_
#         cov[:, :, i] = np.cov(embedding_vectors[:, :, i].numpy(), rowvar=False) + 0.01 * I
        
    
#     return mean, cov
