

import torch
torch.manual_seed(0)
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from model import *
from tqdm import *
from evaluation import *
from datetime import datetime

__all__ = ['loss_fn', 'Trainer']

# calculate graph similarity
class GrfSim(nn.Module):
    def __init__(self, max_num_nodes, device):
        super(GrfSim, self).__init__()
        self.device = device
        self.max_num_nodes= max_num_nodes

    def edge_similarity_matrix(self, adj, adj_recon, matching_features,
                matching_features_recon, sim_func):
        S = torch.zeros(self.max_num_nodes, self.max_num_nodes,
                        self.max_num_nodes, self.max_num_nodes)
        for i in range(self.max_num_nodes):
            for j in range(self.max_num_nodes):
                if i == j:
                    for a in range(self.max_num_nodes):
                        S[i, i, a, a] = adj[i, i] * adj_recon[a, a] * \
                                        sim_func(matching_features[i], matching_features_recon[a])
                else:
                    for a in range(self.max_num_nodes):
                        for b in range(self.max_num_nodes):
                            if b == a:
                                continue
                            S[i, j, a, b] = adj[i, j] * adj[i, i] * adj[j, j] * \
                                            adj_recon[a, b] * adj_recon[a, a] * adj_recon[b, b]
        return S

    def mpm(self, x_init, S, max_iters=50):
        x = x_init
        for it in range(max_iters):
            x_new = torch.zeros(self.max_num_nodes, self.max_num_nodes)
            for i in range(self.max_num_nodes):
                for a in range(self.max_num_nodes):
                    x_new[i, a] = x[i, a] * S[i, i, a, a]
                    pooled = [torch.max(x[j, :] * S[i, j, a, :])
                              for j in range(self.max_num_nodes) if j != i]
                    neigh_sim = sum(pooled)
                    x_new[i, a] += neigh_sim
            norm = torch.norm(x_new)
            x = x_new / norm
        return x

    def deg_feature_similarity(self, f1, f2):
        return 1 / (abs(f1 - f2) + 1)

    def permute_adj(self, adj, curr_ind, target_ind):
        ''' Permute adjacency matrix.
          The target_ind (connectivity) should be permuted to the curr_ind position.
        '''
        # order curr_ind according to target ind
        ind = np.zeros(self.max_num_nodes, dtype=np.int)
        ind[target_ind] = curr_ind
        adj_permuted = torch.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_permuted[:, :] = adj[ind, :]
        adj_permuted[:, :] = adj_permuted[:, ind]
        return adj_permuted

    def adj_recon_loss(self, adj_truth, adj_pred):
        return F.binary_cross_entropy(adj_pred, adj_truth)

    def forward(self, recon_grf, original_grf):
        # set matching features be degree (# of edges): graph matching
        out_features = torch.sum(recon_grf, 1)
        adj_data = original_grf
        adj_features = torch.sum(adj_data, 1)

        S = self.edge_similarity_matrix(adj_data, recon_grf, adj_features, out_features,
                                        self.deg_feature_similarity)

        # initialization strategies
        init_corr = 1 / self.max_num_nodes
        init_assignment = torch.ones(self.max_num_nodes, self.max_num_nodes) * init_corr
        assignment = self.mpm(init_assignment, S)

        adj_permuted = adj_data
        adj_vectorized = adj_permuted[torch.triu(torch.ones(self.max_num_nodes,self.max_num_nodes))== 1].squeeze_()
        adj_vectorized_var = Variable(adj_vectorized)

        recon_permuted= recon_grf

        recon_vectorized = recon_permuted[torch.triu(torch.ones(self.max_num_nodes, self.max_num_nodes)) == 1].squeeze_()
        recon_vectorized_var = Variable(recon_vectorized)
        adj_recon_loss = self.adj_recon_loss(adj_vectorized_var, recon_vectorized_var)

        return adj_recon_loss

def metrics(original_full_adj, recon_adj):
    original_full_adj= original_full_adj.data.cpu().numpy()
    recon_adj= recon_adj.data.cpu().numpy()

    #print("type: ", type(original_adj), type(recon_adj))
    # -------evaluation statistics
    # recon_adj_copy= recon_adj.copy()
    # recon_adj_copy[recon_adj > 0.5] = 1
    # recon_adj_copy[recon_adj <= 0.5] = 0

    #print("original_adj.shape: ", original_adj.shape, "recon_adj.shape: ", recon_adj.shape) #(64, 100, 8, 3)
    # DGs = Discrete_Graphs(original_full_adj)  # compare with the whole dataset
    # FDGs = Discrete_Graphs(recon_adj_copy)
    #print('DGs.graphs', len(DGs.graphs))
    #print('FDGs.graphs', len(FDGs.graphs))

    #Bursty_Coeff= MMD(DGs.Sample_Bursty_Coeff(), FDGs.Sample_Bursty_Coeff())
    #Temporal_Efficiency= MMD(DGs.Sample_Temporal_Degree_Centrality(), FDGs.Sample_Temporal_Degree_Centrality())

    Bursty_Coeff = torch.tensor(0)
    Temporal_Efficiency = torch.tensor(0)
    #Degree_Centrality = np.mean(abs(DGs.Sample_Temporal_Degree_Centrality() - FDGs.Sample_Temporal_Degree_Centrality()))
    Degree_Centrality= torch.tensor(0)

    # a = DGs.Sample_Temporal_Degree_Centrality()
    # b = FDGs.Sample_Temporal_Degree_Centrality()
    # Degree_Centrality= np.mean(np.abs(np.mean(a, axis=0) - np.mean(b, axis=0)))
    return Bursty_Coeff, Temporal_Efficiency, Degree_Centrality


def loss_fn(original_adj, recon_adj, original_feature, recon_feature, f_mean, f_logvar, z_post_mean_edge, z_post_logvar_edge,
                                             z_post_mean_node, z_post_logvar_node, z_post_mean_edge_node, z_post_logvar_edge_node,
                                             z_prior_mean_edge, z_prior_logvar_edge, z_prior_mean_node, z_prior_logvar_node,
                                             z_prior_mean_edge_node, z_prior_logvar_edge_node, original_full_adj, max_num_nodes, device):
    """
    Loss function: 1. The MSE loss between the generated and the original graphs
                   2. The KL divergence of f,
                   3. The sum over the KL divergence of each z_t, with the sum divided by batch_size

    Loss = {mse + KL of f + sum(KL of z_t)} / batch_size
    Prior of f is a spherical zero mean unit variance Gaussian and the prior of each z_t is a Gaussian whose mean and variance
    are given by the LSTM
    """
    batch_size = original_adj.size(0)
    seq_len= original_adj.size(1)
    mse_feature = F.mse_loss(original_feature, recon_feature, reduction='sum')
    Bursty_Coeff, Temporal_Efficiency, Degree_Centrality = metrics(original_full_adj, recon_adj)

    # 1. graphs similarity by mpm (computational expensive)
    # grf_disim= 0
    # grfsim = GrfSim(max_num_nodes, device)
    # grfsim = grfsim.to(device)
    # for i in range(batch_size):
    #     for j in range(seq_len):
    #         loss = grfsim(recon_adj[i][j], original_adj[i][j])
    #         grf_disim+= loss

    # 2. graphs similarity by cross entropy
    original_adj= Variable(original_adj)
    recon_adj= Variable(recon_adj)
    grf_disim= F.binary_cross_entropy(input= recon_adj, target= original_adj)


    original_edge_num= original_adj.sum()/batch_size
    recon_adj_copy= recon_adj.data.cpu().numpy().copy()
    recon_adj_copy[recon_adj_copy > 0.5]= 1
    recon_adj_copy[recon_adj_copy <= 0.5] = 0
    recon_edge_num= recon_adj_copy.sum()/batch_size

    kld_f = -0.5 * torch.sum(1 + f_logvar - torch.pow(f_mean, 2) - torch.exp(f_logvar))

    z_post_var_edge = torch.exp(z_post_logvar_edge)
    z_prior_var_edge = torch.exp(z_prior_logvar_edge)
    z_post_var_node = torch.exp(z_post_logvar_node)
    z_prior_var_node = torch.exp(z_prior_logvar_node)
    z_post_var_edge_node = torch.exp(z_post_logvar_edge_node)
    z_prior_var_edge_node = torch.exp(z_prior_logvar_edge_node)

    kld_z_edge = 0.5 * torch.sum(z_prior_logvar_edge - z_post_logvar_edge +
        ((z_post_var_edge + torch.pow(z_post_mean_edge - z_prior_mean_edge, 2)) / z_prior_var_edge) - 1)
    kld_z_node = 0.5 * torch.sum(z_prior_logvar_node - z_post_logvar_node +
        ((z_post_var_node + torch.pow(z_post_mean_node - z_prior_mean_node, 2)) / z_prior_var_node) - 1)
    kld_z_edge_node = 0.5 * torch.sum(z_prior_logvar_edge_node - z_post_logvar_edge_node +
        ((z_post_var_edge_node + torch.pow(z_post_mean_edge_node - z_prior_mean_edge_node, 2)) / z_prior_var_edge_node) - 1)


    return ((grf_disim + mse_feature) + kld_f + kld_z_edge + kld_z_node+ kld_z_edge_node) / batch_size, \
           grf_disim/batch_size, mse_feature/batch_size,\
           kld_f / batch_size, (kld_z_edge + kld_z_node+ kld_z_edge_node) / batch_size,\
           Bursty_Coeff/ batch_size, Temporal_Efficiency/ batch_size, Degree_Centrality/ batch_size, original_edge_num, recon_edge_num

class Trainer(object):
    def __init__(self, model, trainloader, test_f_expand, max_num_nodes, genr_batch_size, original_full_adj,
                 epochs=3, learning_rate=0.001, nsamples=1, recon_path='./recon/',
                 checkpoints='./output/model.pth', device=torch.device('cuda:0')):
        self.trainloader = trainloader
        self.start_epoch = 0
        self.epochs = epochs
        self.device = device
        self.model = model
        self.model.to(device)
        self.learning_rate = learning_rate
        self.checkpoints = checkpoints
        self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        self.samples = nsamples
        self.sample_path = []
        self.recon_path = recon_path
        self.test_f_expand = test_f_expand
        self.epoch_losses = []

        self.max_num_nodes= max_num_nodes
        self.genr_batch_size= genr_batch_size
        self.original_full_adj= original_full_adj


    def save_checkpoint(self, epoch):
        torch.save({
            'epoch': epoch + 1,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses': self.epoch_losses},
            self.checkpoints)

    def load_checkpoint(self):
        try:
            print("Loading Checkpoint from '{}'".format(self.checkpoints))
            checkpoint = torch.load(self.checkpoints)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_losses = checkpoint['losses']
            print("Resuming Training From Epoch {}".format(self.start_epoch))
        except:
            print("No Checkpoint Exists At '{}'.Start Fresh Training".format(self.checkpoints))
            self.start_epoch = 0

    def sample_graphs(self, epoch):
        with torch.no_grad():
            _, _, test_z_edge = self.model.sample_z(self.genr_batch_size, random_sampling=False)
            _, _, test_z_node = self.model.sample_z(self.genr_batch_size, random_sampling=False)

            print("from sample_frames: edge and node: ", test_z_edge.shape, test_z_node.shape)
            print("from sample_frames: test_f_expand.shape: ", self.test_f_expand.shape)
            test_zf_edge = torch.cat((test_z_edge, self.test_f_expand), dim=2) #fix f -> change
            test_zf_node = torch.cat((test_z_node, self.test_f_expand), dim=2)

            recon_adj, recon_feature = self.model.decode_graphs(test_zf_edge, test_zf_node)
            print("from sample_frames: recon.shape: ", recon_adj.shape, recon_feature.shape)
            recon_adj= recon_adj.cpu()
            recon_feature= recon_feature.cpu()
            np.save('./output/adj_{}.npy'.format("metro_fix_z"), recon_adj)
            np.save('./output/feature_{}.npy'.format("metro_fix_z"), recon_feature)

    def train_model(self):
        self.model.train()
        for epoch in range(self.start_epoch, self.epochs):
            losses = []
            kld_fs = []
            kld_zs = []
            grf_losses = []
            node_losses = []
            Bursty_Coeffs = []
            Temporal_Efficiencies = []
            Degree_Centralities = []
            original_edges = []
            recon_edges = []

            print("Running Epoch : {}".format(epoch + 1))
            for i, dataitem in tqdm(enumerate(self.trainloader, 1)):
                adj, feature = dataitem
                adj= adj.to(self.device)
                feature= feature.to(self.device)
                self.optimizer.zero_grad()
                f_mean, f_logvar, f, z_post_mean_edge, z_post_logvar_edge, z_edge, z_post_mean_node, z_post_logvar_node, z_node, \
                z_mean_edge_node, z_logvar_edge_node, z_edge_node, z_mean_prior_edge, z_logvar_prior_edge, z_mean_prior_node, \
                z_logvar_prior_node, z_mean_prior_edge_node, z_logvar_prior_edge_node, recon_adj, recon_feature = self.model(adj, feature)

                loss, grf_loss, node_loss, kld_f, kld_z, Bursty_Coeff, Temporal_Efficiency, Degree_Centrality, original_edge_num, recon_edge_num = \
                    loss_fn(adj, recon_adj, feature, recon_feature, f_mean, f_logvar, z_post_mean_edge,
                            z_post_logvar_edge, z_post_mean_node, z_post_logvar_node, z_mean_edge_node, z_logvar_edge_node,
                            z_mean_prior_edge, z_logvar_prior_edge, z_mean_prior_node, z_logvar_prior_node,
                            z_mean_prior_edge_node, z_logvar_prior_edge_node, self.original_full_adj, self.max_num_nodes, device=device)

                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                kld_fs.append(kld_f.item())
                kld_zs.append(kld_z.item())

                grf_losses.append(grf_loss.item())
                node_losses.append(node_loss.item())
                Bursty_Coeffs.append(Bursty_Coeff.item())
                Temporal_Efficiencies.append(Temporal_Efficiency.item())
                Degree_Centralities.append(Degree_Centrality.item())
                original_edges.append(original_edge_num.item())
                recon_edges.append(recon_edge_num.item())
            meanloss = np.mean(losses)
            meanf = np.mean(kld_fs)
            meanz = np.mean(kld_zs)

            meangrf_loss = np.mean(grf_losses)
            meannode_losses = np.mean(node_losses)
            meanBursty_Coeffs = np.mean(Bursty_Coeffs)
            meanTemporal_Efficiencies = np.mean(Temporal_Efficiencies)
            meanDegree_Centralities = np.mean(Degree_Centralities)
            original_edges_total= np.sum(original_edges)
            recon_edges_total= np.sum(recon_edges)

            self.epoch_losses.append(meanloss)
            print("Epoch {} : Average Loss: {} Edge Loss {} Node Loss {} KL of f : {} KL of z : {} "
                  "Bursty_Coeff: {} Temporal_Efficiency: {} Degree_Centrality {} original_edges_total {} recon_edges_total {}".
                  format(epoch + 1, meanloss, meangrf_loss,  meannode_losses, meanf, meanz, meanBursty_Coeffs, meanTemporal_Efficiencies, meanDegree_Centralities,
                         original_edges_total, recon_edges_total
                         ))
            self.save_checkpoint(epoch)
            self.model.eval()
            if epoch== self.epochs-1:
            #if epoch%5== 0:
                self.sample_graphs(epoch + 1)
            self.model.train()
        print("Training is complete")


if __name__ == '__main__':

    #--------- load data
    adj = np.load('./dataset/protein_adj.npy')
    features = np.load('./dataset/protein_features.npy')

    #--------- after loading dataset
    # to torch
    adj = torch.from_numpy(adj).float()
    features = torch.from_numpy(features).float()
    dataset = torch.utils.data.TensorDataset(adj, features)

    batch_size = 64
    seq_len = adj.size(1) #length of each sequence
    max_num_nodes= adj.size(2)
    feature_dim= features.size(3)

    genr_batch_size = 100
    f_dim= 256

    loader = torch.utils.data.DataLoader(dataset, batch_size= batch_size, shuffle=True, num_workers=4)
    #device = torch.device('cuda:0')
    device = torch.device('cpu')

    d2g2 = D2G2(f_dim=f_dim, z_dim=32, batch_size= batch_size, seq_len= seq_len, factorised=True, device=device,
                          graphs= seq_len,feature_dim= feature_dim, max_num_nodes= max_num_nodes)

    fix_f= True
    if not fix_f:
        # 1. not fixed f: each snapshot has different f
        test_f = torch.rand(genr_batch_size, f_dim, device=device)
        test_f = test_f.unsqueeze(1).expand(genr_batch_size, seq_len, f_dim)
    else:
        # 2. fixed f : each snapshot has identical f
        fix_f = torch.rand(f_dim, device=device)
        fix_f = fix_f.expand(genr_batch_size, seq_len, f_dim)
        test_f = fix_f


    trainer = Trainer(d2g2, loader, test_f, epochs=100, learning_rate=0.0002,
                      device=device, max_num_nodes= max_num_nodes, genr_batch_size= genr_batch_size, original_full_adj=adj)

    trainer.load_checkpoint()
    
    startime = datetime.now()
    trainer.train_model()
    endtime= datetime.now()
    print("time for training:", endtime-startime)