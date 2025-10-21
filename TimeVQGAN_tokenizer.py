from TimeVQGAN import TimeVQGAN
import torch
from torch import nn
from types import SimpleNamespace
import matplotlib.pyplot as plt


class TimeVQGAN_Tokenizer(nn.Module):
    def __init__(self):
        super(TimeVQGAN_Tokenizer, self).__init__()
        args = SimpleNamespace(
            device='cuda',
            num_codebook_vectors=16384,
            latent_dim=384,
            beta=0.25,
            vqgan_model='checkpoints/TimeVQGAN.pt',
            downsample_ratio=4,
            class_condition=True
        )
        self.TimeVQGAN = TimeVQGAN(args).to(args.device)
        pretrained_dict = {k.replace('module.', ''): v for k, v in
                           torch.load(args.vqgan_model, weights_only=True, map_location='cpu').items()}
        self.TimeVQGAN.load_state_dict(pretrained_dict, strict=True)
        self.TimeVQGAN.eval()



    def unfold_data(self, tensor, batch_len=512):
        num_batches = tensor.size(0) // batch_len
        remainder = tensor.size(0) % batch_len
        padding_length = batch_len - remainder if remainder > 0 else 0
        batches = [tensor[i * batch_len:(i + 1) * batch_len] for i in range(num_batches)]
        if remainder > 0:
            last_batch = tensor[num_batches * batch_len:]
            last_batch = torch.nn.functional.pad(last_batch, (0, padding_length), 'constant', 0)
            batches.append(last_batch)
        batched_tensor = torch.stack(batches) if len(batches) > 0 else tensor.view(1, -1)
        pad_mask = torch.zeros_like(batched_tensor, dtype=torch.bool)
        if remainder > 0 and padding_length > 0:
            pad_mask[-1, -padding_length:] = True
        return batched_tensor, pad_mask

    def fold_data(self, batched_tensor, pad_mask):
        # Ensure mask is on the same device for boolean indexing
        pad_mask = pad_mask.to(batched_tensor.device)
        if batched_tensor.shape[0] == 1:
            original_tensor = batched_tensor[0][~pad_mask[0]]
            return original_tensor
        else:
            unpadded_batches = [batch[~mask] for batch, mask in zip(batched_tensor, pad_mask)]
            original_tensor = torch.cat(unpadded_batches, dim=0)
            return original_tensor


    def encode(self, data):
        data, self.pad_mask = self.unfold_data(data)
        code_index = self.TimeVQGAN.ts2code(data.unsqueeze(-1))
        code_index = code_index.view(-1)
        return code_index

    def decode(self, code_index):
        code_index = code_index.view(-1, 128)
        data = self.TimeVQGAN.code2ts(code_index)
        data = self.fold_data(data, pad_mask=self.pad_mask)
        return data


if __name__ == '__main__':
    model = TimeVQGAN_Tokenizer().cuda()
    tso = torch.sin(torch.linspace(0, 200, 1024))+torch.randn(1024)*0.01
    tso = tso.cuda()
    code_index = model.encode(tso)
    print(code_index)
    tso_recover = model.decode(code_index)

    fig = plt.figure(figsize=(12, 8))
    plt.subplot(211)
    plt.plot(tso.cpu().detach().numpy())
    plt.title("Ori")

    plt.subplot(212)
    plt.plot(tso_recover.cpu().detach().numpy())
    plt.title("Rec")

    plt.show()
