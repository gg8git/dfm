import copy
import math
from collections import defaultdict

import PIL
import numpy as np
import pandas as pd
import torch, time, os
import wandb
import seaborn as sns
import yaml

sns.set_style('whitegrid')
from matplotlib import pyplot as plt
from torch import optim

from model.dna_models import MLPModel, CNNModel, TransformerModel, DeepFlyBrainModel
from utils.esm import upgrade_state_dict
from utils.flow_utils import DirichletConditionalFlow, expand_simplex, sample_cond_prob_path, simplex_proj, \
    get_wasserstein_dist, update_ema, load_flybrain_designed_seqs
from lightning_modules.general_module import GeneralModule
from utils.logging import get_logger


logger = get_logger(__name__)


class MoleculeModule(GeneralModule):
    def __init__(self, args, alphabet_size, toy_data):
        super().__init__(args)
        self.load_model(alphabet_size, None)

        self.condflow = DirichletConditionalFlow(K=self.model.alphabet_size, alpha_spacing=0.001, alpha_max=args.alpha_max)
        self.crossent_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.toy_data = toy_data

        self.val_outputs = defaultdict(list)
        self.train_outputs = defaultdict(list)
        self.train_out_initialized = False
        self.loaded_classifiers = False
        self.loaded_distill_model = False
        self.mean_log_ema = {}
        if self.args.taskiran_seq_path is not None:
            self.taskiran_fly_seqs = load_flybrain_designed_seqs(self.args.taskiran_seq_path).to(self.device)


    def on_load_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = {k: v for k,v in checkpoint['state_dict'].items() if 'cls_model' not in k and 'distill_model' not in k}

    def training_step(self, batch, batch_idx):
        self.stage = 'train'
        loss = self.general_step(batch, batch_idx)
        if self.args.ckpt_iterations is not None and self.trainer.global_step in self.args.ckpt_iterations:
            self.trainer.save_checkpoint(os.path.join(os.environ["MODEL_DIR"],f"epoch={self.trainer.current_epoch}-step={self.trainer.global_step}.ckpt"))
        self.try_print_log()
        return loss

    def validation_step(self, batch, batch_idx):
        self.stage = 'val'
        loss = self.general_step(batch, batch_idx)
        if self.args.validate:
            self.try_print_log()

    def general_step(self, batch, batch_idx=None):
        self.iter_step += 1
        seq = batch
        B, L = seq.shape
        if seq.min().item() < 0 or seq.max().item() > self.model.alphabet_size:
            import ipdb; ipdb.set_trace()
            print("bounds error")

        xt, alphas = sample_cond_prob_path(self.args, seq, self.model.alphabet_size) # [512,49,22], [512]
        if self.args.mode == 'distill':
            if self.stage == 'val':
                seq_distill = torch.zeros_like(seq, device=self.device)
            else:
                logits_distill, xt = self.dirichlet_flow_inference(seq, model=self.distill_model, args=self.distill_args)
                seq_distill = torch.argmax(logits_distill, dim=-1)
            alphas = alphas * 0
        xt_inp = xt
        if self.args.mode == 'dirichlet' or self.args.mode == 'riemannian':
            xt_inp, prior_weights = expand_simplex(xt, alphas, self.args.prior_pseudocount) # [512,49,44], [512,1,1]
            self.lg('prior_weight', prior_weights)

        logits = self.model(xt_inp, t=alphas, cls=None) # [512,49,22]

        losses = torch.nn.functional.cross_entropy(logits.transpose(1, 2), seq_distill if self.args.mode == 'distill' else seq, reduction='none')
        losses = losses.mean(-1)

        self.lg('loss', losses)
        self.lg('perplexity', torch.exp(losses.mean())[None].expand(B))
        if self.stage == "val":
            if self.args.mode == 'dirichlet':
                logits_pred, _ = self.dirichlet_flow_inference(seq, model=self.model, args=self.args)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.args.mode == 'riemannian':
                logits_pred = self.riemannian_flow_inference(seq)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            elif self.args.mode == 'ardm' or self.args.mode == 'lrar':
                seq_pred = self.ar_inference(seq)
            elif self.args.mode == 'distill':
                logits_pred = self.distill_inference(seq)
                seq_pred = torch.argmax(logits_pred, dim=-1)
            else:
                raise NotImplementedError()

            self.lg('seq', [''.join([['A','C','G','T'][num] if self.model.alphabet_size == 4 else str(num) for num in seq]) for seq in seq_pred])
            self.lg('recovery', seq_pred.eq(seq).float().mean(-1))
            if self.args.dataset_type == 'toy_fixed':
                self.log_data_similarities(seq_pred)

            self.val_outputs['seqs'].append(seq_pred.cpu())
        self.lg('alpha', alphas)
        self.lg('dur', torch.tensor(time.time() - self.last_log_time)[None].expand(B))
        self.last_log_time = time.time()
        return losses.mean()

    @torch.no_grad()
    def distill_inference(self, seq):
        B, L = seq.shape
        K = self.model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, K, device=seq.device)).sample()
        logits = self.model(x0, t=torch.zeros(B, device=self.device))
        return logits

    @torch.no_grad()
    def dirichlet_flow_inference(self, seq, model, args):
        B, L = seq.shape
        K = model.alphabet_size
        x0 = torch.distributions.Dirichlet(torch.ones(B, L, model.alphabet_size, device=seq.device)).sample()
        eye = torch.eye(K).to(x0)
        xt = x0.clone()

        t_span = torch.linspace(1, args.alpha_max, self.args.num_integration_steps, device=self.device)
        for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])):
            xt_expanded, _ = expand_simplex(xt, s[None].expand(B), args.prior_pseudocount)
            logits = model(xt_expanded, t=s[None].expand(B))
            flow_probs = torch.nn.functional.softmax(logits / args.flow_temp, -1) # [B, L, K]

            if not torch.allclose(flow_probs.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (flow_probs >= 0).all():
                print(f'WARNING: flow_probs.min(): {flow_probs.min()}. Some values of flow_probs do not lie on the simplex. There are we are {(flow_probs<0).sum()} negative values in flow_probs of shape {flow_probs.shape} that are negative. We are projecting them onto the simplex.')
                flow_probs = simplex_proj(flow_probs)

            c_factor = self.condflow.c_factor(xt.cpu().numpy(), s.item())
            c_factor = torch.from_numpy(c_factor).to(xt)

            self.inf_counter += 1
            if torch.isnan(c_factor).any():
                print(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')
                if self.args.allow_nan_cfactor:
                    c_factor = torch.nan_to_num(c_factor)
                    self.nan_inf_counter += 1
                    if self.nan_inf_counter > 100:
                        raise RuntimeError(f'nan_inf_counter: {self.nan_inf_counter}')
                else:
                    raise RuntimeError(f'NAN cfactor after: xt.min(): {xt.min()}, flow_probs.min(): {flow_probs.min()}')

            if not (flow_probs >= 0).all(): print(f'flow_probs.min(): {flow_probs.min()}')
            cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
            flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

            xt = xt + flow * (t - s)

            if not torch.allclose(xt.sum(2), torch.ones((B, L), device=self.device), atol=1e-4) or not (xt >= 0).all():
                print(f'WARNING: xt.min(): {xt.min()}. Some values of xt do not lie on the simplex. There are we are {(xt<0).sum()} negative values in xt of shape {xt.shape} that are negative. We are projecting them onto the simplex.')
                xt = simplex_proj(xt)
        return logits, x0

    @torch.no_grad()
    def riemannian_flow_inference(self, seq):
        B, L = seq.shape
        K = self.model.alphabet_size
        xt = torch.distributions.Dirichlet(torch.ones(B, L, K)).sample().to(self.device)
        eye = torch.eye(K).to(self.device)

        t_span = torch.linspace(0, 1, self.args.num_integration_steps, device=self.device)
        for s, t in zip(t_span[:-1], t_span[1:]):
            xt_expanded, prior_weights = expand_simplex(xt, s[None].expand(B), self.args.prior_pseudocount)
            logits = self.model(xt_expanded, s[None].expand(B))
            probs = torch.nn.functional.softmax(logits, -1)
            cond_flows = (eye - xt.unsqueeze(-1)) / (1 - s)
            flow = (probs.unsqueeze(-2) * cond_flows).sum(-1)
            xt = xt + flow * (t - s)
        return xt

    @torch.no_grad()
    def ar_inference(self, seq):
        B, L = seq.shape
        order = np.arange(L)
        if self.args.mode == 'ardm': np.random.shuffle(order)
        curr = (torch.ones((B, L), device=self.device) * 4).long()
        for i, k in enumerate(order):
            t = torch.tensor(i / L, device=self.device)
            logits = self.model(torch.nn.functional.one_hot(curr, num_classes=5).float(), t[None].expand(B))
            curr[:, k] = torch.distributions.Categorical(
                probs=torch.nn.functional.softmax(logits[:, k] / self.args.flow_temp, -1)).sample()
        return curr

    def on_validation_epoch_start(self):
        self.inf_counter = 1
        self.nan_inf_counter = 0

    def on_validation_epoch_end(self):
        self.generator = np.random.default_rng()
        log = self._log
        log = {key: log[key] for key in log if "val_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update({'val_nan_inf_step_fraction': self.nan_inf_counter / self.inf_counter})

        mean_log.update({'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})
        if self.args.dataset_type == 'toy_sampled':
            all_seqs = torch.cat(self.val_outputs['seqs'], dim=0).cpu()
            all_seqs_one_hot = torch.nn.functional.one_hot(all_seqs, num_classes=self.args.toy_simplex_dim)
            counts = all_seqs_one_hot.sum(0).float()
            empirical_dist = counts / counts.sum(dim=-1, keepdim=True)
            kl = (empirical_dist * (torch.log(empirical_dist) - torch.log(self.toy_data.probs[self.args.target_class]))).sum(-1).mean()
            rkl = (self.toy_data.probs[self.args.target_class] * (torch.log(self.toy_data.probs[self.args.target_class]) - torch.log(empirical_dist))).sum(-1).mean()
            sanity_self_kl = (empirical_dist * (torch.log(empirical_dist) - torch.log(empirical_dist))).sum(-1).mean()
            mean_log.update({'val_kl': kl.cpu().item(), 'val_rkl': rkl.cpu().item(), 'val_sanity_self_kl': sanity_self_kl.cpu().item()})

        self.mean_log_ema = update_ema(current_dict=mean_log, prev_ema=self.mean_log_ema, gamma=0.9)
        mean_log.update(self.mean_log_ema)
        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                wandb.log(mean_log)
                if self.args.dataset_type == 'toy_sampled':
                    pil_dist_comp = self.plot_empirical_and_true(empirical_dist, self.toy_data.probs[self.args.target_class])
                    wandb.log({'fig': [wandb.Image(pil_dist_comp)], 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

            path = os.path.join(os.environ["MODEL_DIR"], f"val_{self.trainer.global_step}.csv")
            pd.DataFrame(log).to_csv(path)

        for key in list(log.keys()):
            if "val_" in key:
                del self._log[key]
        self.val_outputs = defaultdict(list)

    def on_train_epoch_start(self) -> None:
        self.inf_counter = 1
        self.nan_inf_counter = 0
        if not self.loaded_distill_model and self.args.distill_ckpt is not None:
            self.load_distill_model()
            self.loaded_distill_model = True

    def on_train_epoch_end(self):
        self.train_out_initialized = True
        log = self._log
        log = {key: log[key] for key in log if "train_" in key}
        log = self.gather_log(log, self.trainer.world_size)
        mean_log = self.get_log_mean(log)
        mean_log.update(
            {'epoch': float(self.trainer.current_epoch), 'step': float(self.trainer.global_step), 'iter_step': float(self.iter_step)})

        if self.trainer.is_global_zero:
            logger.info(str(mean_log))
            self.log_dict(mean_log, batch_size=1)
            if self.args.wandb:
                wandb.log(mean_log)

        for key in list(log.keys()):
            if "train_" in key:
                del self._log[key]

    def lg(self, key, data):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        log = self._log
        if self.args.validate or self.stage == 'train':
            log["iter_" + key].extend(data)
        log[self.stage + "_" + key].extend(data)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def plot_empirical_and_true(self, empirical_dist, true_dist):
        num_datasets_to_plot = min(4, empirical_dist.shape[0])
        width = 1
        # Creating a figure and axes
        fig, axes = plt.subplots(math.ceil(num_datasets_to_plot/2), 2, figsize=(10, 8))
        for i in range(num_datasets_to_plot):
            row, col = i // 2, i % 2
            x = np.arange(len(empirical_dist[i]))
            axes[row, col].bar(x, empirical_dist[i], width, label=f'empirical')
            axes[row, col].plot(x, true_dist[i], label=f'true density', color='orange')
            axes[row, col].legend()
            axes[row, col].set_title(f'Sequence position {i + 1}')
            axes[row, col].set_xlabel('Category')
            axes[row, col].set_ylabel('Density')
        plt.tight_layout()
        fig.canvas.draw()
        pil_img = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        plt.close()
        return pil_img

    def load_model(self, alphabet_size, num_cls):
        if self.args.model == 'cnn':
            self.model = CNNModel(self.args, alphabet_size=alphabet_size, num_cls=num_cls)
        elif self.args.model == 'mlp':
            self.model = MLPModel(self.args, alphabet_size=alphabet_size, num_cls=num_cls)
        elif self.args.model == 'transformer':
            self.model = TransformerModel(self.args, alphabet_size=alphabet_size, num_cls=num_cls)
        elif self.args.model == 'deepflybrain':
            self.model = DeepFlyBrainModel(self.args, alphabet_size=alphabet_size,num_cls=num_cls)
        else:
            raise NotImplementedError()

    def load_distill_model(self):
        with open(self.args.distill_ckpt_hparams) as f:
            hparams = yaml.load(f, Loader=yaml.UnsafeLoader)
            self.distill_args = copy.deepcopy(hparams['args'])
        if self.distill_args.model == 'cnn':
            self.distill_model = CNNModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'mlp':
            self.distill_model = MLPModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'transformer':
            self.distill_model = TransformerModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        elif self.distill_args.model == 'deepflybrain':
            self.distill_model = DeepFlyBrainModel(self.distill_args, alphabet_size=self.model.alphabet_size,num_cls=self.model.num_cls)
        else:
            raise NotImplementedError()
        upgraded_dict = upgrade_state_dict(torch.load(self.args.distill_ckpt, map_location=self.device)['state_dict'], prefixes=['model.'])
        no_cls_dict = {k: v for k, v in upgraded_dict.items() if 'cls_model' not in k}
        self.distill_model.load_state_dict(no_cls_dict)
        self.distill_model.eval()
        self.distill_model.to(self.device)
        for param in self.distill_model.parameters():
            param.requires_grad = False

    def plot_score_and_probs(self):
        clss = torch.cat(self.val_outputs['clss_noisycls'])
        probs = torch.softmax(torch.cat(self.val_outputs['logits_noisycls']), dim=-1)
        scores = torch.cat(self.val_outputs['scores_noisycls']).cpu().numpy()
        score_norms = np.linalg.norm(scores, axis=-1)
        alphas = torch.cat(self.val_outputs['alphas_noisycls']).cpu().numpy()
        true_probs = probs[torch.arange(len(probs)), clss].cpu().numpy()
        bins = np.linspace(min(alphas), 12, 20)
        indices = np.digitize(alphas, bins)
        bin_means = [np.mean(true_probs[indices == i]) for i in range(1, len(bins))]
        bin_std = [np.std(true_probs[indices == i]) for i in range(1, len(bins))]
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        bin_pos_std = [np.std(true_probs[indices == i][true_probs[indices == i] > np.mean(true_probs[indices == i])]) for i in range(1, len(bins))]
        bin_neg_std = [np.std(true_probs[indices == i][true_probs[indices == i] < np.mean(true_probs[indices == i])]) for i in range(1, len(bins))]
        plot_data = pd.DataFrame({'Alphas': bin_centers, 'Means': bin_means, 'Std': bin_std, 'Pos_Std': bin_pos_std, 'Neg_Std': bin_neg_std})
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Alphas', y='Means', data=plot_data)
        plt.fill_between(plot_data['Alphas'], plot_data['Means'] - plot_data['Neg_Std'], plot_data['Means'] + plot_data['Pos_Std'], alpha=0.3)
        plt.xlabel('Binned alphas values')
        plt.ylabel('Mean of predicted probs for true class')
        fig = plt.gcf()
        fig.canvas.draw()
        pil_probs = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

        plt.close()
        bin_means = [np.mean(score_norms[indices == i]) for i in range(1, len(bins))]
        bin_std = [np.std(score_norms[indices == i]) for i in range(1, len(bins))]
        bin_pos_std = [np.std(score_norms[indices == i][score_norms[indices == i] > np.mean(score_norms[indices == i])]) for i in range(1, len(bins))]
        bin_neg_std = [np.std(score_norms[indices == i][score_norms[indices == i] < np.mean(score_norms[indices == i])]) for i in range(1, len(bins))]
        plot_data = pd.DataFrame({'Alphas': bin_centers, 'Means': bin_means, 'Std': bin_std, 'Pos_Std': bin_pos_std, 'Neg_Std': bin_neg_std})
        plt.figure(figsize=(10, 6))
        sns.lineplot(x='Alphas', y='Means', data=plot_data)
        plt.fill_between(plot_data['Alphas'], plot_data['Means'] - plot_data['Neg_Std'],
                         plot_data['Means'] + plot_data['Pos_Std'], alpha=0.3)
        plt.xlabel('Binned alphas values')
        plt.ylabel('Mean of norm of the scores')
        fig = plt.gcf()
        fig.canvas.draw()
        pil_score_norms = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        return pil_probs, pil_score_norms

    def log_data_similarities(self, seq_pred):
        similarities1 = seq_pred.cpu()[:, None, :].eq(self.toy_data.data_class1[None, :, :])  # batchsize, dataset_size, seq_len
        similarities2 = seq_pred.cpu()[:, None, :].eq(self.toy_data.data_class2[None, :, :])  # batchsize, dataset_size, seq_len
        similarities = seq_pred.cpu()[:, None, :].eq(torch.cat([self.toy_data.data_class2[None, :, :], self.toy_data.data_class1[None, :, :]],dim=1))  # batchsize, dataset_size, seq_len
        self.lg('data1_sim', similarities1.float().mean(-1).max(-1)[0])
        self.lg('data2_sim', similarities2.float().mean(-1).max(-1)[0])
        self.lg('data_sim', similarities.float().mean(-1).max(-1)[0])
        self.lg('mean_data1_sim', similarities1.float().mean(-1).mean(-1))
        self.lg('mean_data2_sim', similarities2.float().mean(-1).mean(-1))
        self.lg('mean_data_sim', similarities.float().mean(-1).mean(-1))