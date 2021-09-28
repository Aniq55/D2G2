
import torch
import torch.nn as nn

# A block consisting of convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnit, self).__init__()  # y = max(0, x) + leak*min(0,x)
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
                nn.BatchNorm2d(out_channels), nonlinearity)  # batch normalization
        else:
            self.model = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel, stride, padding), nonlinearity)

    def forward(self, x):
        return self.model(x)

#------------------- GraphConv
# GCN basic operation
# f(Hⁱ, A) = σ(AHⁱWⁱ)
# Each layer Hⁱ corresponds to an N × Fⁱ feature matrix where each row is a feature representation of a node.
#The weight matrix has dimensions Fⁱ × Fⁱ⁺¹，which determines the number of features at the next layer.
class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConv, self).__init__()
        self.input_dim = input_dim  #input feature dimension for node.
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.randn([input_dim, output_dim]), requires_grad=True)

    def forward(self, feature, adj):
        y = torch.matmul(adj, feature)
        y = torch.matmul(y,self.weight)
        return y


#-------------------GrfConvUnit
# A block consisting of graph convolution, batch normalization followed by a nonlinearity
class GrfConvUnit(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, max_num_nodes, pool='sum'):
        '''
        Args:
            input_dim: input feature dimension for node. (equals to max_num_nodes for feature_type == 'id')
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.  in vae？
        '''
        super(GrfConvUnit, self).__init__()
        self.pool= pool
        self.conv1= GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.bn1= nn.BatchNorm1d(max_num_nodes)
        self.act= nn.ReLU()
        self.conv2= GraphConv(input_dim=hidden_dim, output_dim=hidden_dim)
        self.bn2 = nn.BatchNorm1d(max_num_nodes)

    def pool_graph(self, x):
        if self.pool == 'max':
                out, _ = torch.max(x, dim=1, keepdim=False)
        elif self.pool == 'sum':
                out = torch.sum(x, dim=1, keepdim=False)
        return out

    def forward(self, feature, adj):
        x = self.conv1(feature, adj)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x, adj)
        x = self.bn2(x)

        graph_h= self.pool_graph(x)
        return graph_h


# A block consisting of a transposed convolution, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class ConvUnitTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0, out_padding=0, batchnorm=True,
                 nonlinearity=nn.LeakyReLU(0.2)):
        super(ConvUnitTranspose, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding),
                nn.BatchNorm2d(out_channels), nonlinearity)
        else:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel, stride, padding, out_padding), nonlinearity)

    def forward(self, x):
        return self.model(x)


class GrfConvUnitTranspose(nn.Module):
    def __init__(self, max_num_nodes, seq_len, device):
        super(GrfConvUnitTranspose, self).__init__()
        self.max_num_nodes= max_num_nodes
        self.seq_len= seq_len
        self.device = device

    def recover_adj_lower(self, l, batch_size):
        adj = torch.zeros(batch_size, self.seq_len, self.max_num_nodes, self.max_num_nodes, device=self.device)
        adj[torch.triu(torch.ones(batch_size, self.seq_len, self.max_num_nodes, self.max_num_nodes)) == 1] = l
        return adj

    def recover_full_adj_from_lower(self, lower, batch_size):
        lower= lower.view(-1, self.max_num_nodes, self.max_num_nodes)
        triu = torch.triu(torch.ones(batch_size*self.seq_len, self.max_num_nodes, self.max_num_nodes, device=self.device))
        triu = torch.transpose(triu, 1, 2)
        lower_tri = triu - torch.diag_embed(torch.ones(batch_size*self.seq_len, self.max_num_nodes, device=self.device))
        full= lower + torch.transpose(lower, 1, 2) * lower_tri
        # The sequence length is reintroduced
        full= full.view(batch_size, self.seq_len, self.max_num_nodes, self.max_num_nodes)
        return full

    def forward(self, encoded_graph, current_batch_size):
        recon_adj_lower= self.recover_adj_lower(encoded_graph, current_batch_size)
        recon_adj_tensor= self.recover_full_adj_from_lower(recon_adj_lower, current_batch_size)
        return recon_adj_tensor


# A block consisting of an affine layer, batch normalization (optional) followed by a nonlinearity (defaults to Leaky ReLU)
class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                nn.Linear(in_features, out_features), nonlinearity)

    def forward(self, x):
        return self.model(x)


class D2G2(nn.Module):
    """
    Hyperparameters:
        f_dim: Dimension of the content encoding f. f has the shape (batch_size, f_dim)
        z_dim: Dimension of the dynamics encoding of a frame z_t. z has the shape (batch_size, frames, z_dim)
        conv_dim: The convolutional encoder converts each graph into an intermediate encoding vector of size conv_dim.
        hidden_dim: Dimension of the hidden states of the RNNs
        factorised: Toggles between full and factorised posterior for z as discussed in the paper

    Optimization:
        The model is trained with the Adam optimizer with a learning rate of 0.0002, betas of 0.9 and 0.999
    """

    def __init__(self, f_dim=256, z_dim=32, conv_dim=64, hidden_dim=512, graphs=8, factorised=False,
                 device=torch.device('cuda:0'), batch_size= 6, seq_len= 8,
                 feature_dim=16, gcn_dim= 64, max_num_nodes=16):
        super(D2G2, self).__init__()
        self.device = device
        self.f_dim = f_dim  # per the whole dynamic graphs (batch_size, f_dim)
        self.z_dim = z_dim  # per graph of the dynamic graphs (batch_size, graphs, z_dim)
        self.graphs = graphs  # -> # of graphs of one sequence
        self.conv_dim = conv_dim  # encodes each graph: (num_channels,in_size,in_size)=>conv_dim
        self.hidden_dim = hidden_dim  # of the hidden states of the RNNs
        self.factorised = factorised
        self.in_size = feature_dim  # adj matrix dimensions of the graphs
        self.batch_size= batch_size
        self.seq_len= seq_len
        self.feature_dim= feature_dim
        self.gcn_dim= gcn_dim
        self.max_num_nodes = max_num_nodes
        self.output_dim = max_num_nodes * (max_num_nodes + 1) // 2   # make edge prediction (reconstruct)

        # Prior of statics is a uniform Gaussian and prior of the dynamics is an LSTM
        self.z_prior_lstm = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        # Posterior distribution networks
        self.f_lstm = nn.LSTM(self.output_dim+ self.conv_dim, self.hidden_dim, 1,
                              bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        if self.factorised is True:
            self.z_inter_edge = LinearUnit(self.output_dim, self.hidden_dim, batchnorm=False)
            self.z_inter_node = LinearUnit(self.conv_dim, self.hidden_dim, batchnorm=False)
            self.z_inter_edge_node= LinearUnit(self.output_dim+ self.conv_dim, self.hidden_dim, batchnorm=False)

            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)
        else:
            self.z_lstm_edge = nn.LSTM(self.output_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
            self.z_lstm_node = nn.LSTM(self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
            self.z_lstm_edge_node= nn.LSTM(self.output_dim+ self.conv_dim + self.f_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)

            self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
            # Each timestep is for each z so no reshaping and feature mixing
            self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
            self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        #-----------for node/feature-----------
        # ConvUnit(in_channels, out_channels, kernel, stride=1, padding=0, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2))
        #------@@@ 1.  mlp with cnn
        # self.conv_feature = nn.Sequential(
        #     ConvUnit(1, step, 2, 1, 2),  # 3*64*64 -> 256*64*64  3: in_channels  256: out_channels
        #     ConvUnit(step, step, 2, 2, 2),  # 256,64,64 -> 256,32,32
        #     ConvUnit(step, step, 2, 2, 2),  # 256,32,32 -> 256,16,16
        #     ConvUnit(step, step, 2, 2, 2),  # 256,16,16 -> 256,8,8
        # )
        #
        # self.final_conv_size_feature= 4  #in_size=64
        # self.conv_fc_feature = nn.Sequential(
        #     LinearUnit(step * (self.final_conv_size_feature ** 2), self.conv_dim * 2),
        #     LinearUnit(self.conv_dim * 2, self.conv_dim))
        #
        # self.deconv_feature = nn.Sequential(
        #     ConvUnitTranspose(step, step, 2, 2, 2, 1),
        #     ConvUnitTranspose(step, step, 2, 2, 2, 1),
        #     ConvUnitTranspose(step, step, 2, 2, 2, 1),
        #     ConvUnitTranspose(step, 1, 2, 1, 2, 0, nonlinearity=nn.Tanh()))
        #
        # self.deconv_fc_feature = nn.Sequential(
        #     LinearUnit(self.f_dim + self.z_dim, self.conv_dim * 2, False),
        #     LinearUnit(self.conv_dim * 2, step * (self.final_conv_size_feature ** 2), False))

        # ------@@@ 2. mlp without cnn
        self.conv_fc_feature = nn.Sequential(
            LinearUnit(self.max_num_nodes*self.feature_dim, self.conv_dim * 2),
            LinearUnit(self.conv_dim * 2, self.conv_dim))

        self.deconv_fc_feature = nn.Sequential(
            LinearUnit(self.f_dim + self.z_dim + self.z_dim, self.conv_dim * 2, False),
            LinearUnit(self.conv_dim * 2, self.max_num_nodes*self.feature_dim, False))


        # ----------- for edge/adj-----------
        self.grfConv= GrfConvUnit(input_dim=self.feature_dim, hidden_dim=self.gcn_dim,
                                  latent_dim= 256, max_num_nodes=self.max_num_nodes)

        self.final_conv_size= self.gcn_dim
        self.conv_fc = nn.Sequential(
            LinearUnit(self.final_conv_size, self.conv_dim * 2),
            LinearUnit(self.conv_dim * 2, self.output_dim))  # fc: fully connected

        self.grfDeconv= GrfConvUnitTranspose(self.max_num_nodes, self.seq_len, device=device)

        self.deconv_fc = nn.Sequential(
            LinearUnit(self.f_dim + self.z_dim + self.z_dim, self.conv_dim * 2, False),
            LinearUnit(self.conv_dim * 2, self.output_dim, False))


        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    # If random sampling is true, reparametrization occurs else z_t is just set to the mean
    def sample_z(self, genr_batch_size, random_sampling=True):
        z_out = None  # This will ultimately store all z_s in the format [batch_size, graphs, z_dim]
        z_means = None
        z_logvars = None

        # All states are initially set to 0, especially z_0 = 0
        z_t = torch.zeros(genr_batch_size, self.z_dim, device=self.device)
        h_t = torch.zeros(genr_batch_size, self.hidden_dim, device=self.device)
        c_t = torch.zeros(genr_batch_size, self.hidden_dim, device=self.device)

        fixed_z = False
        if not fixed_z:
            #print("varying z")
            for _ in range(self.graphs):
                h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
                z_mean_t = self.z_prior_mean(h_t)
                z_logvar_t = self.z_prior_logvar(h_t)
                z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)

                if z_out is None:
                    # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                    z_out = z_t.unsqueeze(1)
                    z_means = z_mean_t.unsqueeze(1)
                    z_logvars = z_logvar_t.unsqueeze(1)
                else:
                    # If z_out is not none, z_t is not the initial z and hence append it to the previous z_ts collected in z_out
                    z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)  # concatenate by dim=1
                    z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                    z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        else:
            #print("z is fixed")
            h_t, c_t = self.z_prior_lstm(z_t, (h_t, c_t))
            z_mean_t = self.z_prior_mean(h_t)
            z_logvar_t = self.z_prior_logvar(h_t)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            z_out= z_t.unsqueeze(1).expand(genr_batch_size, self.graphs, self.z_dim)
            z_means = z_mean_t.unsqueeze(1).expand(genr_batch_size, self.graphs, self.z_dim)
            z_logvars = z_logvar_t.unsqueeze(1).expand(genr_batch_size, self.graphs, self.z_dim)

        return z_means, z_logvars, z_out


    def encode_graphs(self, feature, adj):
        #--------encode edge--------
        # The feature and adj are unrolled into the batch dimension for batch processing such that go from
        # [batch_size, graphs, size, size] to [batch_size * graphs, size, size]
        feature= feature.view(-1, self.max_num_nodes, self.feature_dim)
        adj= adj.view(-1, self.max_num_nodes, self.max_num_nodes)
        graph_h= self.grfConv(feature, adj)
        graph_h= self.conv_fc(graph_h)
        # The graph dimension is reintroduced and the shape becomes [batch_size, graphs, conv_dim]
        graph_h= graph_h.view(-1, self.graphs, self.output_dim)

        # --------encode feature with cnn and mlp--------
        # #print("x.shape:", feature.shape) #[192, 8, 8]
        # feature = feature.view(-1, 1, self.feature_dim, self.feature_dim)  #input: [N(batch size), C(channel), H(height), W(width)]
        # feature_h = self.conv_feature(feature)
        # #print("feature_h:", feature_h.shape)  #[192, 256, 4, 4]
        # feature_h = feature_h.view(-1, self.step * (self.final_conv_size_feature ** 2))
        # feature_h = self.conv_fc_feature(feature_h)
        # # The frame dimension is reintroduced and x shape becomes [batch_size, frames, conv_dim]
        # # This technique is repeated at several points in the code
        # feature_h = feature_h.view(-1, self.graphs, self.conv_dim)
        # #print("feature_h:", feature_h.shape, feature_h)  #[64, 3, 64]

        # --------encode feature with mlp ONLY--------
        feature = feature.view(-1, self.max_num_nodes*self.feature_dim)
        feature_h= self.conv_fc_feature(feature)
        feature_h = feature_h.view(-1, self.graphs, self.conv_dim)

        return graph_h, feature_h


    def decode_graphs(self, zf_edge, zf_node):
        # --------decode edge--------
        decoded_graph= self.deconv_fc(zf_edge)
        current_batch_size= decoded_graph.size(0)
        decoded_graph= decoded_graph.view(-1, current_batch_size*self.seq_len*self.output_dim) # flattened as input
        decoded_graph= self.grfDeconv(decoded_graph, current_batch_size)
        decoded_graph= decoded_graph.view(-1, self.seq_len, self.max_num_nodes, self.max_num_nodes)
        decoded_graph = torch.sigmoid(decoded_graph)  # [10, 3, 8, 8]

        # # --------decode feature with cnn and mlp--------
        # decoded_feature= self.deconv_fc_feature(zf_node)
        # decoded_feature= decoded_feature.view(-1, self.step, self.final_conv_size_feature, self.final_conv_size_feature)
        # decoded_feature= self.deconv_feature(decoded_feature)
        # decoded_feature = decoded_feature.view(-1, self.graphs, self.feature_dim, self.feature_dim)
        # #print("decoded_feature.shape: ", decoded_feature.shape)  #[30, 3, 1, 8, 8]

        # --------decode feature with mlp ONLY--------
        decoded_feature = self.deconv_fc_feature(zf_node)
        decoded_feature = decoded_feature.view(-1, self.graphs, self.max_num_nodes, self.feature_dim)
        #print("decoded_feature.shape: ", decoded_feature.shape)  # [10, 3, 8, 2]

        return decoded_graph, decoded_feature


    def reparameterize(self, mean, logvar, random_sampling=True):
        # Reparametrization occurs only if random sampling is set to true, otherwise mean is returned
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def encode_f(self, x):
        lstm_out, _ = self.f_lstm(x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.graphs - 1, 0:self.hidden_dim]
        lstm_out = torch.cat((frontal, backward), dim=1)
        mean = self.f_mean(lstm_out)
        logvar = self.f_logvar(lstm_out)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z_node(self, node, f):
        if self.factorised is True:
            features = self.z_inter_node(node)
        else:
            # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
            f_expand = f.unsqueeze(1).expand(-1, self.graphs, self.f_dim)
            lstm_out, _ = self.z_lstm_node(torch.cat((node, f_expand), dim=2))
            features, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(features)
        logvar = self.z_logvar(features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z_edge(self, edge, f):
        if self.factorised is True:
            edges = self.z_inter_edge(edge)
        else:
            # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
            f_expand = f.unsqueeze(1).expand(-1, self.graphs, self.f_dim)
            lstm_out, _ = self.z_lstm_edge(torch.cat((edge, f_expand), dim=2))
            edges, _ = self.z_rnn(lstm_out)
        mean = self.z_mean(edges)
        logvar = self.z_logvar(edges)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)

    def encode_z_edge_node(self, x, f):
        if self.factorised is True:
            edges_features= self.z_inter_edge_node(x)
        else:
            # The expansion is done to match the dimension of x and f, used for concatenating f to each x_t
            f_expand = f.unsqueeze(1).expand(-1, self.graphs, self.f_dim)
            lstm_out, _ = self.z_lstm_edge_node(torch.cat((x, f_expand), dim= 2))
            edges_features, _ = self.z_rnn(lstm_out)
        mean= self.z_mean(edges_features)
        logvar= self.z_logvar(edges_features)
        return mean, logvar, self.reparameterize(mean, logvar, self.training)



    def forward(self, adj, feature):
        z_mean_prior_edge, z_logvar_prior_edge, _ = self.sample_z(adj.size(0), random_sampling=self.training)
        z_mean_prior_node, z_logvar_prior_node, _ = self.sample_z(feature.size(0), random_sampling=self.training)
        z_mean_prior_edge_node, z_logvar_prior_edge_node, _ = self.sample_z(feature.size(0), random_sampling=self.training)

        conv_edge, conv_node= self.encode_graphs(feature, adj)
        conv_edge_node= torch.cat((conv_edge, conv_node), dim= 2)
        f_mean, f_logvar, f = self.encode_f(conv_edge_node)


        z_mean_edge, z_logvar_edge, z_edge = self.encode_z_edge(conv_edge, f)
        z_mean_node, z_logvar_node, z_node = self.encode_z_node(conv_node, f)
        z_mean_edge_node, z_logvar_edge_node, z_edge_node = self.encode_z_edge_node(conv_edge_node, f)


        f_expand = f.unsqueeze(1).expand(-1, self.graphs, self.f_dim)
        zf_edge = torch.cat((z_edge, z_edge_node, f_expand), dim=2)
        zf_node = torch.cat((z_node, z_edge_node, f_expand), dim=2)

        recon_adj, recon_feature = self.decode_graphs(zf_edge, zf_node)

        return f_mean, f_logvar, f, z_mean_edge, z_logvar_edge, z_edge, z_mean_node, z_logvar_node, z_node,\
               z_mean_edge_node, z_logvar_edge_node, z_edge_node,\
               z_mean_prior_edge, z_logvar_prior_edge, z_mean_prior_node, z_logvar_prior_node, \
               z_mean_prior_edge_node, z_logvar_prior_edge_node, recon_adj, recon_feature
