"""Script para o treinamento de uma rede de segmentação. As únicas
modificações deste script em relação ao de classificação são:

1. o uso do parâmetro ignore_index=2 na classe CrossEntropyLoss.
Ele faz com que os pixeis com valor 2 na imagem de rótulos sejam ignorados.

2. inclusão da função collate_fn no dataloader de validação, pois
as imagens não possuem o mesmo tamanho.

3. Modificação da função de acurácia para medir a intersecção sobre a união.
"""

# Gambiarra para importar o script train.py feito anteriormente
import sys

import torch
from src.dataset import collate_fn, get_dataset
from torch import nn
import numpy as np
import random
import matplotlib.pyplot as plt
from IPython import display
from torch.utils.data import DataLoader
import wandb
import os
import torchvision.transforms.v2 as transf


def seed_all(seed):
    """
    Set random seed for PyTorch, NumPy, and Python random.
    Args:
        seed (int): Seed value.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def show_log(logger, save_path=None):
    """
    Plot training and validation metrics in a notebook and optionally save the figure.
    Args:
        logger (list): List of tuples (epoch, train_loss, valid_loss, perf).
        save_path (str, optional): Path to save the figure.
    """
    epochs, losses_train, losses_valid, accs = zip(*logger)
    ious = [perf for _, _, _, perf in logger]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    ax1.plot(epochs, losses_train, "-o", ms=2, label="Train loss")
    ax1.plot(epochs, losses_valid, "-o", ms=2, label="Valid loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_ylim((0, 1.0))
    ax1.legend()
    # Plot acurácia e IoU no mesmo subplot
    # ax2.plot(epochs, accs, "-o", ms=2, color="tab:blue", label="Accuracy")
    ax2.plot(epochs, ious, "-o", ms=2, color="tab:orange", label="IoU")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score")
    ax2.set_ylim((0, 1.0))
    ax2.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path)
    display.clear_output(wait=True)
    plt.show()
    plt.close(fig)


def train_step(model, dl_train, optim, loss_func, scheduler, device):
    """
    Perform one training epoch.
    Args:
        model (nn.Module): Model to train.
        dl_train (DataLoader): Training data loader.
        optim (Optimizer): Optimizer.
        loss_func (callable): Loss function.
        scheduler (Scheduler): Learning rate scheduler.
        device (str): Device to use.
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    loss_log = 0.0
    for imgs, targets in dl_train:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optim.zero_grad()
        scores = model(imgs)
        loss = loss_func(scores, targets)
        loss.backward()
        optim.step()

        # Multiplica por imgs.shape[0] porque o último batch pode ter tamanho diferente
        loss_log += loss.detach() * imgs.shape[0]

    # Muda o learning rate
    scheduler.step()

    # Média das losses calculadas
    loss_log /= len(dl_train.dataset)

    return loss_log.item()


# Anotador para evitar que gradientes sejam registrados dentro da função
@torch.no_grad()
def valid_step(model, dl_valid, loss_func, perf_func, device):
    """
    Perform one validation epoch.
    Args:
        model (nn.Module): Model to validate.
        dl_valid (DataLoader): Validation data loader.
        loss_func (callable): Loss function.
        perf_func (callable): Performance metric function.
        device (str): Device to use.
    Returns:
        tuple: (average validation loss, average performance metric)
    """
    model.eval()
    loss_log = 0.0
    perf_log = 0.0
    for imgs, targets in dl_valid:
        imgs = imgs.to(device)
        targets = targets.to(device)
        scores = model(imgs)
        loss = loss_func(scores, targets)
        perf = perf_func(scores, targets)

        # Multiplica por imgs.shape[0] porque o último batch pode ter tamanho diferente
        loss_log += loss * imgs.shape[0]
        perf_log += perf * imgs.shape[0]

    # Média dos valores calculados
    loss_log /= len(dl_valid.dataset)
    perf_log /= len(dl_valid.dataset)

    return loss_log.item(), perf_log.item()


@torch.no_grad()
def iou(scores, targets):
    """
    Calculate Intersection over Union (IoU) between prediction and ground truth.
    Args:
        scores (torch.Tensor): Model output scores.
        targets (torch.Tensor): Ground truth mask.
    Returns:
        float: IoU value.
    """
    # Transforma a predição da rede em índices 0 e 1, e aplica em reshape
    # nos tensores para transformá-los em 1D
    pred = scores.argmax(dim=1).reshape(-1)
    targets = targets.reshape(-1)

    # Mantém apenas valores para os quais target!=2. O valor 2 indica píxeis
    # a serem ignorados
    pred = pred[targets != 2]
    targets = targets[targets != 2]

    # Verdadeiro positivos
    tp = ((targets == 1) & (pred == 1)).sum()
    # Verdadeiro negativos
    tn = ((targets == 0) & (pred == 0)).sum()
    # Falso positivos
    fp = ((targets == 0) & (pred == 1)).sum()
    # Falso negativos
    fn = ((targets == 1) & (pred == 0)).sum()

    # Algumas métricas interessantes para medir a qualidade do resultado
    # Fração de píxeis corretos
    _ = (tp + tn) / (tp + tn + fp + fn)
    # Intersecção sobre a união (IoU)
    iou = tp / (tp + fp + fn)
    # Precisão
    _ = tp / (tp + fp)
    # Revocação
    _ = tp / (tp + fn)

    # Retorna apenas o iou para não termos que reescrever a função de plotagem
    # dos resultados, que espera um único valor de performance
    return iou


def setup_wandb(args, checkpoint_dir):
    """
    Configure and initialize Weights & Biases (wandb).
    Args:
        args (object): Arguments object with wandb settings.
        checkpoint_dir (str): Directory to save checkpoints and wandb files.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Inicializa o wandb
    wandb.init(
        project=args.wandb_project,
        group=args.wandb_group,
        name=args.run_name,
        config={
            "learning_rate": args.lr,
            "epochs": args.num_epochs,
            "batch_size": args.bs_train,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
        },
        # Salva os arquivos do notebook no diretório do wandb
        dir=checkpoint_dir,
        # Ativa o monitoramento de recursos
        reinit=True,
    )

    # Adiciona informações adicionais se fornecidas
    if args.meta is not None:
        wandb.run.log({"meta": args.meta})


def log_wandb_samples(model, ds_valid, indices, device, epoch, prefix="val_sample"):
    """
    Log validation images to wandb: original, ground truth, and segmentation.
    Args:
        model (nn.Module): Trained model.
        ds_valid (Dataset): Validation dataset.
        indices (list): List of sample indices to log.
        device (str): Device to use.
        epoch (int): Current epoch.
        prefix (str, optional): Prefix for wandb log names.
    """
    import torchvision.transforms.functional as F

    model.eval()
    images = []
    for idx in indices:
        img, target = ds_valid[idx]
        img_tensor = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = pred.argmax(dim=1).cpu().squeeze().numpy()
        # Converte para PIL para logar no wandb
        img_pil = F.to_pil_image(img.cpu())
        gt_pil = F.to_pil_image(target.cpu().byte())
        pred_pil = F.to_pil_image(torch.from_numpy(pred_mask).byte())
        images.append(
            [
                wandb.Image(img_pil, caption=f"{prefix}_{idx}_input"),
                wandb.Image(gt_pil, caption=f"{prefix}_{idx}_gt"),
                wandb.Image(pred_pil, caption=f"{prefix}_{idx}_pred"),
            ]
        )
    # Loga como tabela
    columns = ["Input", "GroundTruth", "Prediction"]
    wandb.log(
        {f"{prefix}_epoch_{epoch}": wandb.Table(data=images, columns=columns)},
        step=epoch,
    )

def train(
    model,
    bs_train,
    bs_valid,
    num_epochs,
    lr,
    weight_decay=0.0,
    resize_size=224,
    seed=0,
    num_workers=5,
    checkpoint_dir="../data/checkpoints/M07",
    data_root="./data/oxford_pets",
    use_wandb=False,
    wandb_project=None,
    wandb_group=None,
    meta=None,
    run_name=None,
    experiment_name=None,
    log_val_samples=False,
    val_sample_indices=None,
    # Adiciona os parâmetros de arquitetura para salvar no checkpoint
    use_strides=None,
    extra_convs=None,
):
    """
    Train a segmentation model for multiple epochs.
    Args:
        model (nn.Module): Model to train.
        bs_train (int): Batch size for training.
        bs_valid (int): Batch size for validation.
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.
        weight_decay (float, optional): Weight decay.
        resize_size (int, optional): Resize size for images.
        seed (int, optional): Random seed.
        num_workers (int, optional): Number of DataLoader workers.
        checkpoint_dir (str, optional): Directory for checkpoints.
        data_root (str, optional): Dataset root directory.
        use_wandb (bool, optional): Whether to use wandb.
        wandb_project (str, optional): wandb project name.
        wandb_group (str, optional): wandb group name.
        meta (any, optional): Additional metadata for wandb.
        run_name (str, optional): wandb run name.
        experiment_name (str, optional): Experiment name.
        log_val_samples (bool, optional): Log validation samples to wandb.
        val_sample_indices (list, optional): Indices of validation samples to log.
        use_strides (list, optional): Strides for decoder.
        extra_convs (bool, optional): Use extra conv layers in decoder.
    Returns:
        tuple: (ds_train, ds_valid, logger)
            ds_train (Dataset): Training dataset.
            ds_valid (Dataset): Validation dataset.
            logger (list): Training log with metrics per epoch.
    """
    seed_all(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds_train, ds_valid, class_weights = get_dataset(data_root, resize_size=resize_size)
    # ds_train.indices = ds_train.indices[:5*256]
    model.to(device)

    dl_train = DataLoader(
        ds_train,
        batch_size=bs_train,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    # parâmetro collate_fn é necessário porque as imagens de validação possuem
    # tamanhos distintos
    dl_valid = DataLoader(
        ds_valid,
        batch_size=bs_valid,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )

    # ignore_index=2 indica que os pixeis com valor 2 na imagem de rótulos serão ignorados.
    # Isso inclui pixeis de borda e pixeis utilizados no padding da função collate_fn
    # acima
    loss_func = nn.CrossEntropyLoss(
        torch.tensor(class_weights, device=device), ignore_index=2
    )
    optim = torch.optim.SGD(
        model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
    )
    sched = torch.optim.lr_scheduler.PolynomialLR(optim, num_epochs)
    logger = []
    best_loss = torch.inf
    # Setup wandb se solicitado
    if use_wandb:

        class Args:
            pass

        args = Args()
        args.wandb_project = wandb_project
        args.experiment_name = experiment_name if experiment_name else run_name
        args.run_name = run_name
        args.wandb_group = wandb_group
        args.meta = meta
        # Adiciona os hiperparâmetros necessários
        args.lr = lr
        args.num_epochs = num_epochs
        args.bs_train = bs_train
        args.weight_decay = weight_decay
        args.seed = seed
        setup_wandb(args, checkpoint_dir)

    for epoch in range(0, num_epochs):
        loss_train = train_step(model, dl_train, optim, loss_func, sched, device)
        loss_valid, iou_value = valid_step(model, dl_valid, loss_func, iou, device)
        logger.append((epoch, loss_train, loss_valid, iou_value))

        # Salva o gráfico de log sobrescrevendo a mesma imagem
        log_plot_path = f"{checkpoint_dir}/log.png"
        show_log(logger, save_path=log_plot_path)

        # Loga métricas no wandb
        if use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "lr": sched.get_last_lr()[0],
                    "train_loss": loss_train,
                    "valid_loss": loss_valid,
                    "iou": iou_value,
                },
                step=epoch,
            )
            # Loga amostras do set de validação, se solicitado
            if log_val_samples and val_sample_indices is not None and epoch % 1 == 0:
                log_wandb_samples(model, ds_valid, val_sample_indices, device, epoch)
        # Dados sobre o estado atual
        checkpoint = {
            "params": {
                "bs_train": bs_train,
                "bs_valid": bs_valid,
                "lr": lr,
                "weight_decay": weight_decay,
                "use_strides": use_strides,
                "extra_convs": extra_convs,
            },
            "model": model.state_dict(),
            "optim": optim.state_dict(),
            "sched": sched.state_dict(),
            "logger": logger,
        }

        # Salva o estado atual
        torch.save(checkpoint, f"{checkpoint_dir}/checkpoint.pt")

        # Melhor modelo encontrado
        if loss_valid < best_loss:
            torch.save(checkpoint, f"{checkpoint_dir}/best_model.pt")
    
            best_loss = loss_valid

    model.to("cpu")

    return ds_train, ds_valid, logger
