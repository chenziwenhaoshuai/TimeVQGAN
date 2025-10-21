from time_moe.models.modeling_time_moe import TimeMoeModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize


class TimeEncoder(nn.Module):
    def __init__(self, model_path='./TimeMoE_50M', pretrain_weights='TimeMoE-pretrain.pt', load_pretrain=False, device='cuda'):
        super(TimeEncoder, self).__init__()
        self.down_projector = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=4, padding=0)
        self.Time_encoder = TimeMoeModel.from_pretrained(model_path, torch_dtype=torch.float32)
        if load_pretrain:
            self.Time_encoder.load_state_dict(torch.load(pretrain_weights,weights_only=True))
        freeze_layer = [0, 1, 2, 3, 4, 5]
        for key, param in self.Time_encoder.named_parameters():
            if 'layers.'in key:
                if int(key.split('.')[1]) in freeze_layer:
                    param.requires_grad = False


    def forward(self, ts):
        ts = self.down_projector(ts.permute(0, 2, 1)).permute(0, 2, 1)
        attention_mask = torch.ones(ts.size(0), ts.size(1)).cuda()
        position_ids = torch.arange(ts.size(1)).unsqueeze(0).expand(ts.size(0), -1).cuda()
        output = self.Time_encoder(ts, attention_mask=attention_mask, position_ids=position_ids)
        ts_embedding = output.last_hidden_state
        hidden_states = output.hidden_states

        return ts_embedding, hidden_states


class TimeDecoder(nn.Module):
    def __init__(self, input_dim=384, upsample_rate=4, num_layers=12, nhead=8, dim_feedforward=1024):
        super(TimeDecoder,self).__init__()
        self.upsample_rate = int(upsample_rate-1)
        self.upsample_stages = int(upsample_rate)

        self.transformer_blocks = nn.ModuleList()
        for _ in range(num_layers):
            encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward,
                                                       batch_first=True)
            self.transformer_blocks.append(nn.TransformerEncoder(encoder_layer, num_layers=1))

        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1)

        self.upsample_layers = nn.ModuleList()
        for _ in range(self.upsample_stages):
            self.upsample_layers.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))

        self.out_proj = nn.Linear(input_dim, 1)

    def forward(self, x):
        up_layer = [5, 7, 9, 11]
        j = 0
        for i, transformer in enumerate(self.transformer_blocks):
            x = transformer(x)
            if i in up_layer[:self.upsample_rate]:
                x = x.permute(0, 2, 1)
                x = self.conv1d(x)
                x = self.upsample_layers[j](x)
                x = x.permute(0, 2, 1)
                j += 1

        x = self.out_proj(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, channels=1, num_filters_last=64, n_layers=3):
        super(Discriminator, self).__init__()

        layers = [
            nn.Conv1d(channels, num_filters_last, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        num_filters_mult = 1

        for i in range(1, n_layers):
            num_filters_mult_last = num_filters_mult
            num_filters_mult = min(2 ** i, 8)
            layers += [
                nn.Conv1d(num_filters_last * num_filters_mult_last,
                         num_filters_last * num_filters_mult,
                         4, 2, 1, bias=False),
                nn.BatchNorm1d(num_filters_last * num_filters_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        layers.append(nn.Conv1d(num_filters_last * num_filters_mult, 1, 16, 1, 0))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.model(x)

class Codebook(nn.Module):
    def __init__(self,num_codebook_vectors=256, latent_dim=384, beta=0.25):
        super(Codebook, self).__init__()
        self.num_codebook_vectors = num_codebook_vectors
        self.latent_dim = latent_dim
        self.beta = beta
        self.vq = VectorQuantize(
            dim=latent_dim,
            codebook_size=num_codebook_vectors,
            decay=0.8,
            commitment_weight=beta,
            threshold_ema_dead_code=2,
        )

    def forward(self, z):
        z = z.permute(0, 2, 1).contiguous()
        quantized, indices, loss = self.vq(z)
        quantized = quantized.permute(0, 2, 1)
        return quantized, indices, loss


class NetLinLayer(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super(NetLinLayer, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(),
            nn.Conv1d(in_channels, out_channels, 1, 1, 0, bias=False)
        )

class LPIPS(nn.Module):
    def __init__(self):
        super(LPIPS, self).__init__()
        self.lin0 = NetLinLayer(384)
        self.lin1 = NetLinLayer(384)
        self.lin2 = NetLinLayer(384)
        self.lin3 = NetLinLayer(384)
        self.lin4 = NetLinLayer(384)
        self.init_weights()
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight, 0, 1)

    def norm_tensor(self, x):
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + 1e-10)

    def forward(self, feat_real, feat_fake):
        compute_layer = [1, 2, 3, 4, 5]
        diffs = []
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for i in compute_layer:
            feat_real_i = feat_real[i]
            feat_fake_i = feat_fake[i]
            diffs.append((self.norm_tensor(feat_real_i) - self.norm_tensor(feat_fake_i))**2)
        results = []
        for j in range(5):
            weight = lins[j].model(diffs[j].permute(0, 2, 1))
            pool = F.avg_pool1d(weight, weight.size(-1))
            results.append(pool)
        return torch.mean(torch.cat(results, dim=1))
class TimeVQGAN(nn.Module):
    def __init__(self, args):
        device = args.device
        super(TimeVQGAN, self).__init__()
        self.encoder = TimeEncoder().to(device=device)
        self.decoder = TimeDecoder(upsample_rate=3).to(device=device)
        self.codebook = Codebook(num_codebook_vectors=args.num_codebook_vectors, latent_dim=args.latent_dim, beta=args.beta).to(device=device)
        self.quant_conv = nn.Conv1d(384, 384, 1).to(device=device)
        self.post_quant_conv = nn.Conv1d(384, 384, 1).to(device=device)
        self.LPIPS = LPIPS().to(device=device)

    def forward(self, ts):
        encoded_ts, hidden_states = self.encoder(ts)
        quant_conv_encoded_ts = self.quant_conv(encoded_ts.permute(0, 2, 1))
        quant_conv_encoded_ts = F.normalize(quant_conv_encoded_ts, dim=1)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_ts)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_ts = self.decoder(post_quant_conv_mapping.permute(0, 2, 1))
        with torch.no_grad():
            _, decoded_hidden_states = self.encoder(decoded_ts)
        perceptual_loss = self.LPIPS(hidden_states, decoded_hidden_states)

        return decoded_ts, codebook_indices, q_loss, perceptual_loss

    def ts2code(self,ts):
        encoded_ts, hidden_states = self.encoder(ts)
        quant_conv_encoded_ts = self.quant_conv(encoded_ts.permute(0, 2, 1))
        quant_conv_encoded_ts = F.normalize(quant_conv_encoded_ts, dim=1)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quant_conv_encoded_ts)
        return codebook_indices

    def code2ts(self,codebook_indices):
        codebook_mapping = self.codebook.vq.get_output_from_indices(codebook_indices).permute(0, 2, 1)
        post_quant_conv_mapping = self.post_quant_conv(codebook_mapping)
        decoded_ts = self.decoder(post_quant_conv_mapping.permute(0, 2, 1))
        return decoded_ts

    def calculate_lambda(self, perceptual_loss, gan_loss):
        last_layer = self.decoder.out_proj
        last_layer_weight = last_layer.weight
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        lda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)
        lda = torch.clamp(lda, 0, 1e4).detach()
        return 0.8 * lda

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor
