from torch.utils.data import WeightedRandomSampler, Subset

from lightning_modules.molecule_module import MoleculeModule
from utils.dataset import ToyDataset, TwoClassOverfitDataset, EnhancerDataset, KmerDataModule, MoleculeDataModule
from utils.parsing import parse_train_args
args = parse_train_args()
import torch, os, wandb
torch.manual_seed(0)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import scipy
import torchdiffeq
from utils.flow_utils import sample_cond_prob_path, expand_simplex, simplex_proj

if args.wandb:
    wandb.init(
        entity="molformers",
        settings=wandb.Settings(start_method="fork"),
        project="dfm",
        name=args.run_name,
        config=args,
    )

try:
    anon_login = os.getlogin() == 'anonymized'
except OSError:
    anon_login = False

trainer = pl.Trainer(
    default_root_dir=os.environ["MODEL_DIR"],
    accelerator="gpu" if torch.cuda.is_available() else 'auto',
    max_steps=args.max_steps,
    max_epochs=args.max_epochs,
    num_sanity_val_steps=0,
    limit_train_batches=args.limit_train_batches,
    limit_val_batches=args.limit_val_batches,
    enable_progress_bar=not (args.wandb or args.no_tqdm) or anon_login,
    gradient_clip_val=args.grad_clip,
    callbacks=[
        ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"],
            save_top_k=5,
            save_last=True,
            monitor='val_fxd_generated_to_allseqs' if args.fid_early_stop else 'val_perplexity',
            mode = "min"
        )
    ],
    check_val_every_n_epoch=args.check_val_every_n_epoch,
    val_check_interval=args.val_check_interval,
)


if args.dataset_type == 'toy_fixed':
    train_ds = TwoClassOverfitDataset(args)
    val_ds = train_ds
    toy_data = train_ds
elif args.dataset_type == 'toy_sampled':
    train_ds = ToyDataset(args)
    val_ds = train_ds
    toy_data = train_ds
elif args.dataset_type == 'enhancer':
    train_ds = EnhancerDataset(args, split='train')
    val_ds = EnhancerDataset(args, split='valid' if not args.validate_on_test else 'test')
    toy_data = None

if args.dataset_type == 'kmers':
    print('kmer dataset')
    dm = KmerDataModule(batch_size=args.batch_size, k=1, version=1) # can incorporate args
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader() if not args.validate_on_test else dm.test_dataloader()
    toy_data = None
if args.dataset_type == 'selfies':
    print('selfies dataset')
    dm = MoleculeDataModule(batch_size=args.batch_size, k=1, version=1) # can incorporate args
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader() if not args.validate_on_test else dm.test_dataloader()
    toy_data = None
else:
    if args.subset_train_as_val:
        val_set_size = len(val_ds) if args.constant_val_len is None else args.constant_val_len
        val_ds = Subset(train_ds, torch.randperm(len(train_ds))[:val_set_size])

    if args.oversample_target_class:
        weights = torch.zeros(len(train_ds))
        is_target_cls = train_ds.clss == args.target_class
        weights[is_target_cls] = 0.5 / is_target_cls.sum()
        weights[~is_target_cls] = 0.5 / (~is_target_cls).sum()
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(train_ds), replacement=True)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.dataset_type == 'enhancer')
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.num_workers)

checkpoint_path = os.path.join("workdir", "train_molecule_dfm_final_2_2025-04-24_07-15-18/epoch=89-step=221940.ckpt")
model = MoleculeModule.load_from_checkpoint(checkpoint_path)
torch.save(model.state_dict(), "../NFBO/model_weight/dfm_model.pt")
# model = MoleculeModule() 
# model.load_state_dict(torch.load("../NFBO/model_weight/dfm_model.pt"))
model.eval()
import ipdb; ipdb.set_trace()

def compute_flow(xt, s, device):
    B, L, K = xt.shape
    eye = torch.eye(K).to(device)

    if not torch.allclose(xt.sum(2), torch.ones((B, L), device=device), atol=1e-4) or not (xt >= 0).all():
        xt = simplex_proj(xt)

    xt_expanded, _ = expand_simplex(xt, s[None].expand(B), args.prior_pseudocount)
    logits = model.model.forward(xt_expanded, t=s[None].expand(B))
    flow_probs = torch.nn.functional.softmax(logits / args.flow_temp, -1) # [B, L, K]

    if not torch.allclose(flow_probs.sum(2), torch.ones((B, L), device=device), atol=1e-4) or not (flow_probs >= 0).all():
        flow_probs = simplex_proj(flow_probs)

    c_factor = model.condflow.c_factor(xt.detach().cpu().numpy(), s.item())
    c_factor = torch.from_numpy(c_factor).to(xt)

    # changed
    c_factor = torch.nan_to_num(c_factor, nan=0.0, posinf=1e6, neginf=-1e6)
    c_factor = torch.clamp(c_factor, min=1e-6)

    cond_flows = (eye - xt.unsqueeze(-1)) * c_factor.unsqueeze(-2)
    flow = (flow_probs.unsqueeze(-2) * cond_flows).sum(-1)

    flow = torch.nan_to_num(flow, nan=0.0, posinf=1e6, neginf=-1e6)
    flow = torch.clamp(flow, min=-1e6, max=1e6)

    return flow.to(device)

for seqs in train_loader:
    # model.training_step((seqs, _), batch_idx=None)

    K = dm.full_dataset.alphabet_size
    B, L = seqs.shape

    # import ipdb; ipdb.set_trace()

    # batch_acc_linear = 0
    # batch_acc_ode = 0
    # for seq in seqs:
    #     seq = seq.unsqueeze(0)
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     # target_logits
    #     target_logits = torch.nn.functional.one_hot(seq, num_classes=K).float().to(device)
    #     perturbed_logits = torch.distributions.Dirichlet(target_logits * 100 + 1e-1).sample()
        
    #     # encoding
    #     encode_traj = torchdiffeq.odeint(
    #         lambda t, x: compute_flow(x, t, device),
    #         perturbed_logits.to(device),  # target output
    #         torch.linspace(args.alpha_max, 1, steps=args.num_integration_steps).to(device), # idk what this should be
    #         atol=1e-4, # stay same
    #         rtol=1e-4, # stay same
    #         method="dopri5", # stay same
    #     )

    #     x0 = encode_traj[-1]

    #     # decoding linear
    #     nan_inf_counter = 0
    #     eye = torch.eye(K).to(x0)
    #     xt = x0.clone()
    #     t_span = torch.linspace(1, args.alpha_max, args.num_integration_steps, device=device)
    #     for i, (s, t) in enumerate(zip(t_span[:-1], t_span[1:])): # iter, curr time, next time
    #         flow = compute_flow(xt, s, device)

    #         xt = xt + flow * (t - s)

    #     if not torch.allclose(xt.sum(2), torch.ones((1, L), device=device), atol=1e-4) or not (xt >= 0).all():
    #         xt = simplex_proj(xt)

    #     xt_expanded, _ = expand_simplex(xt, t_span[-1].expand(1), args.prior_pseudocount)
    #     logits = model.model.forward(xt_expanded, t=t_span[-1].expand(1))
    #     norm_logits_linear = torch.nn.functional.softmax(logits / args.flow_temp, -1)

    #     # decoding ode
    #     decode_traj = torchdiffeq.odeint(
    #         lambda t, x: compute_flow(x, t, device),
    #         x0.to(device),
    #         torch.linspace(1, args.alpha_max, steps=args.num_integration_steps).to(device), 
    #         atol=1e-4, 
    #         rtol=1e-4, 
    #         method="dopri5",
    #     )

    #     xt = decode_traj[-1]
    #     norm_logits_ode = torch.nn.functional.softmax(xt, -1)

    #     # compare to target_logits
    #     losses_linear = torch.nn.functional.cross_entropy(norm_logits_linear.view(-1, K).to(device), target_logits.view(-1, K).to(device), reduction='none').mean(-1)
    #     losses_ode = torch.nn.functional.cross_entropy(norm_logits_ode.view(-1, K).to(device), target_logits.view(-1, K).to(device), reduction='none').mean(-1)

    #     pred_linear = norm_logits_linear.argmax(dim=-1)
    #     pred_ode = norm_logits_ode.argmax(dim=-1)
    #     pred_acc_linear = torch.sum(pred_linear.to(device) == seq.to(device)) / seq.numel()
    #     pred_acc_ode = torch.sum(pred_ode.to(device) == seq.to(device)) / seq.numel()

    #     batch_acc_linear += pred_acc_linear
    #     batch_acc_ode += pred_acc_ode

    #     print("next datapoint")

    import ipdb; ipdb.set_trace()
    
    print('start from xt')
    stacked_seqs = seqs.view(-1, 8, L)
    batch_acc_ode = 0
    for seq in stacked_seqs:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # target_logits
        target_logits = torch.nn.functional.one_hot(seq, num_classes=K).float().to(device)
        perturbed_logits = torch.distributions.Dirichlet(target_logits * 100 + 1e-1).sample()
        
        # encoding
        encode_traj = torchdiffeq.odeint(
            lambda t, x: compute_flow(x, t, device),
            perturbed_logits.to(device),  # target output
            torch.linspace(args.alpha_max, 1, steps=args.num_integration_steps).to(device), # idk what this should be
            atol=1e-5, # stay same
            rtol=1e-5, # stay same
            method="dopri5", # stay same
        )

        x0 = encode_traj[-1]

        # decoding
        decode_traj = torchdiffeq.odeint(
            lambda t, x: compute_flow(x, t, device),
            x0.to(device),
            torch.linspace(1, args.alpha_max, steps=args.num_integration_steps).to(device), 
            atol=1e-5, 
            rtol=1e-5, 
            method="dopri5",
        )

        xt = decode_traj[-1]
        norm_logits_ode = torch.nn.functional.softmax(xt, -1)

        # compare to target_logits
        losses_ode = torch.nn.functional.cross_entropy(norm_logits_ode.view(-1, K).to(device), target_logits.view(-1, K).to(device), reduction='none').mean(-1)
        pred_ode = norm_logits_ode.argmax(dim=-1)
        pred_acc_ode = torch.sum(pred_ode.to(device) == seq.to(device)) / seq.numel()

        batch_acc_ode += pred_acc_ode

        print("next minibatch")
    
    print(f'start from xt accuracy: {batch_acc_ode.item()} / {len(stacked_seqs)}')
    
    import ipdb; ipdb.set_trace()
    
    print('start from x0')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x0s = torch.distributions.Dirichlet(torch.ones(B, L, K, device=device)).sample()
    stacked_x0s = x0s.view(-1, 8, L, K)
    batch_acc_ode = 0
    for x0 in stacked_x0s:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # decoding
        decode_traj = torchdiffeq.odeint(
            lambda t, x: compute_flow(x, t, device),
            x0.to(device),
            torch.linspace(1, args.alpha_max, steps=args.num_integration_steps).to(device), 
            atol=1e-4, 
            rtol=1e-4,
            method="dopri5",
        )

        xt = decode_traj[-1]
        norm_logits_ode = torch.nn.functional.softmax(xt, -1)

        # pred seq
        seq_ode = norm_logits_ode.argmax(dim=-1)

        # target_logits
        target_logits = torch.nn.functional.one_hot(seq_ode, num_classes=K).float().to(device)
        perturbed_logits = torch.distributions.Dirichlet(target_logits * 100 + 1e-1).sample()
        
        # encoding
        encode_traj = torchdiffeq.odeint(
            lambda t, x: compute_flow(x, t, device),
            perturbed_logits.to(device),  # target output
            torch.linspace(args.alpha_max, 1, steps=args.num_integration_steps).to(device), # idk what this should be
            atol=1e-4, # stay same
            rtol=1e-4, # stay same
            method="dopri5", # stay same
        )

        x0 = encode_traj[-1]

        # decoding
        decode_traj = torchdiffeq.odeint(
            lambda t, x: compute_flow(x, t, device),
            x0.to(device),
            torch.linspace(1, args.alpha_max, steps=args.num_integration_steps).to(device), 
            atol=1e-4, 
            rtol=1e-4, 
            method="dopri5",
        )

        xt = decode_traj[-1]
        norm_logits_ode = torch.nn.functional.softmax(xt, -1)

        # compare to target_logits
        losses_ode = torch.nn.functional.cross_entropy(norm_logits_ode.view(-1, K).to(device), target_logits.view(-1, K).to(device), reduction='none').mean(-1)

        pred_ode = norm_logits_ode.argmax(dim=-1)
        pred_acc_ode = torch.sum(pred_ode.to(device) == seq_ode.to(device)) / seq_ode.numel()

        batch_acc_ode += pred_acc_ode

        print("next minibatch")

    print(f'start from x0 accuracy: {batch_acc_ode.item()} / {len(stacked_x0s)}')
    
    import ipdb; ipdb.set_trace()

    print("next batch")

    

