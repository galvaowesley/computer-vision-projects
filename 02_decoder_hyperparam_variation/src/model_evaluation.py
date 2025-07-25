
# Função para "desnormalizar" imagens normalizadas pelo padrão ImageNet

from tqdm import tqdm

# Imports no topo do arquivo
import os
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import EncoderDecoder
from src.dataset import get_dataset
from torchvision.models import resnet50, ResNet50_Weights

def build_encoder():
    encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    """
    Build a ResNet50 encoder with pretrained weights and freeze its parameters.
    Returns:
        nn.Module: Frozen ResNet50 encoder.
    """
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder

def filter_model_args(params):
    # Garante que os parâmetros de arquitetura estejam presentes, com valores padrão robustos
    args = {}
    args["decoder_channels"] = 64
    args["num_classes"] = 2
    """
    Ensure architecture parameters are present, with robust default values.
    Args:
        params (dict): Dictionary of model parameters.
    Returns:
        dict: Filtered and completed model arguments.
    """
    # Parâmetros de arquitetura, com fallback para valores padrão robustos
    use_strides = params.get("use_strides")
    extra_convs = params.get("extra_convs")
    if not use_strides:
        use_strides = [32]
    if extra_convs is None:
        extra_convs = False
    args["use_strides"] = use_strides
    args["extra_convs"] = extra_convs
    return args

def denormalize(img):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    """
    Denormalize an image tensor using ImageNet mean and std.
    Args:
        img (torch.Tensor): Normalized image tensor (C, H, W).
    Returns:
        np.ndarray: Denormalized image array (C, H, W).
    """
    img = img.cpu().numpy()
    img = (img * std[:, None, None]) + mean[:, None, None]
    img = np.clip(img, 0, 1)
    return img

# Função auxiliar para carregar modelos a partir da lista de experiments
def load_models_from_experiments(checkpoint_dir, experiments, device="cuda"):
    models = []
    for exp in experiments:
        model_name = exp["name"]
        model_args = filter_model_args(exp["params"])
        ckpt_path = os.path.join(checkpoint_dir, model_name, "best_model.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        encoder = build_encoder()
        model = EncoderDecoder(resnet_encoder=encoder, **model_args)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        models.append(model)
    return models
    """
    Load models from experiment configurations and checkpoints.
    Args:
        checkpoint_dir (str): Directory containing model checkpoints.
        experiments (list): List of experiment dicts with 'name' and 'params'.
        device (str, optional): Device to load models on.
    Returns:
        list: List of loaded EncoderDecoder models.
    """

def iou(scores, targets):
    """Função que calcula a Intersecção sobre a União entre o resultado
    da rede e o rótulo conhecido.
    """
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


# 1. Performance: IoU médio e desvio padrão para cada modelo

def evaluate_models_performance(checkpoint_dir, data_root, experiments, device="cuda"):
    results = []
    ds_train, ds_test, class_weights = get_dataset(data_root, split=0.2)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)
    models = load_models_from_experiments(checkpoint_dir, experiments, device)
    """
    Evaluate segmentation models on test set and collect performance metrics.
    Args:
        checkpoint_dir (str): Directory containing model checkpoints.
        data_root (str): Root directory of the dataset.
        experiments (list): List of experiment dicts.
        device (str, optional): Device to run models on.
    Returns:
        pd.DataFrame: DataFrame with model performance metrics.
    """
    for exp, model in tqdm(zip(experiments, models), total=len(experiments), desc="Avaliando modelos"):
        model_name = exp["name"]
        n_params = sum(p.numel() for p in model.parameters())
        iou_scores = []
        start_time = time.time()
        with torch.no_grad():
            for img, target in dl_test:
                img = img.to(device)
                target = target.to(device)
                pred = model(img)
                iou_value = iou(pred, target)
                iou_scores.append(iou_value.item() if hasattr(iou_value, 'item') else float(iou_value))
        end_time = time.time()
        total_inference_time = end_time - start_time
        avg_inference_time = total_inference_time / len(dl_test)
        mean_iou = np.mean(iou_scores)
        std_iou = np.std(iou_scores)
        results.append({
            "model": model_name,
            "mean_iou": mean_iou,
            "std_iou": std_iou,
            "n_params": n_params,
            "total_inference_time_s": total_inference_time,
            "avg_inference_time_s": avg_inference_time,
            "dataset_size": len(ds_test)
        })
    df = pd.DataFrame(results)
    return df

# 2. Qualidade Visual: plot comparativo das segmentações

def plot_model_segmentations(checkpoint_dir, data_root, experiments, indices, device="cuda", save_pdf=False, pdf_path="segmentations.pdf"):
    ds_train, ds_test, class_weights = get_dataset(data_root, split=0.2)
    models = load_models_from_experiments(checkpoint_dir, experiments, device)
    model_names = [exp["name"] for exp in experiments]
    n_models = len(models)
    n_imgs = len(indices)
    fig, axs = plt.subplots(n_imgs, 2 + n_models, figsize=(4*(2+n_models), 4*n_imgs))
    for row, idx in enumerate(indices):
        img, gt = ds_test[idx]
        axs[row,0].imshow(denormalize(img).transpose(1,2,0))
        axs[row,0].set_title(f"Imagem [{idx}]", fontsize=20)
        axs[row,0].axis('off')
        axs[row,1].imshow(gt.cpu().numpy(), cmap='gray')
        axs[row,1].set_title("Padrão ouro", fontsize=20)
        axs[row,1].axis('off')
        for col, model in enumerate(models):
            with torch.no_grad():
                pred = model(img.unsqueeze(0).to(device))
                pred_mask = pred.argmax(dim=1).cpu().squeeze().numpy()
                iou_value = iou(pred, gt.to(device))
            axs[row,2+col].imshow(pred_mask, cmap='gray')
            axs[row,2+col].set_title(f"{model_names[col]}\nIoU: {iou_value:.3f}", fontsize=20)
            axs[row,2+col].axis('off')
    if save_pdf:
        fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    """
    Plot comparative segmentations for selected models and images.
    Args:
        checkpoint_dir (str): Directory containing model checkpoints.
        data_root (str): Root directory of the dataset.
        experiments (list): List of experiment dicts.
        indices (list): List of image indices to plot.
        device (str, optional): Device to run models on.
        save_pdf (bool, optional): Whether to save the plot as PDF.
        pdf_path (str, optional): Path to save PDF file.
    """


# Plota grid de imagens e ground truths, com opção de salvar em PDF
def plot_image_and_gt_grid(data_root, indices, split="test", save_pdf=False, pdf_path="image_gt_grid.pdf"):
    """
    Plota um grid com imagens e seus ground truths.
    - data_root: caminho do dataset
    - indices: lista de índices das imagens
    - split: "train" ou "test"
    - save_pdf: se True, salva o grid em pdf
    - pdf_path: caminho do arquivo pdf
    """
    """
    Plot a grid of images and their ground truths.
    Args:
        data_root (str): Path to the dataset.
        indices (list): List of image indices.
        split (str, optional): "train" or "test" split.
        save_pdf (bool, optional): Whether to save the grid as PDF.
        pdf_path (str, optional): Path to save PDF file.
    """
    ds_train, ds_test, _ = get_dataset(data_root, split=0.2)
    ds = ds_test if split == "test" else ds_train
    n_imgs = len(indices)
    fig, axs = plt.subplots(2, n_imgs, figsize=(4*n_imgs, 8))
    for col, idx in enumerate(indices):
        img, gt = ds[idx]
        axs[0, col].imshow(denormalize(img).transpose(1,2,0))
        axs[0, col].set_title(f"Imagem [{idx}]")
        axs[0, col].axis('off')
        axs[1, col].imshow(gt.cpu().numpy(), cmap='gray')
        axs[1, col].set_title("Padrão ouro", fontsize=20)
        axs[1, col].axis('off')
    plt.tight_layout()
    if save_pdf:
        plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()
