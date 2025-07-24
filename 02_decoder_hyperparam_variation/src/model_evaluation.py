import os
import torch
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import EncoderDecoder
from src.dataset import get_dataset
from torchvision.utils import make_grid
from torchvision.models import resnet50, ResNet50_Weights

def build_encoder():
    encoder = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder

def filter_model_args(params):
    # Garante que os parâmetros de arquitetura estejam presentes, com valores padrão robustos
    args = {}
    args["decoder_channels"] = 64
    args["num_classes"] = 2
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

# 1. Performance: IoU médio e desvio padrão para cada modelo

def evaluate_models_performance(checkpoint_dir, data_root, models_configs, device="cuda"):
    results = []
    ds_train, ds_test, class_weights = get_dataset(data_root, split=0.2)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=1, shuffle=False)
    for config in models_configs:
        model_name = config["name"]
        ckpt_path = os.path.join(checkpoint_dir, model_name, "best_model.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Usa os parâmetros do experimento, não do checkpoint
        model_args = filter_model_args(config)
        encoder = build_encoder()
        print(f"Instanciando modelo {model_name} com args: {model_args}")
        model = EncoderDecoder(resnet_encoder=encoder, **model_args)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        # Quantidade de parâmetros do modelo
        n_params = sum(p.numel() for p in model.parameters())

        iou_scores = []
        start_time = time.time()
        with torch.no_grad():
            for img, target in dl_test:
                img = img.to(device)
                target = target.to(device)
                pred = model(img)
                pred_mask = pred.argmax(dim=1)
                # IoU
                intersection = ((pred_mask == 1) & (target == 1)).sum().item()
                union = ((pred_mask == 1) | (target == 1)).sum().item()
                iou = intersection / union if union > 0 else 0.0
                iou_scores.append(iou)
        end_time = time.time()
        total_inference_time = end_time - start_time
        # O tempo médio é o tempo total dividido pelo número de amostras
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

def plot_model_segmentations(checkpoint_dir, data_root, model_names, indices, device="cuda"):
    ds_train, ds_test, class_weights = get_dataset(data_root, split=0.2)
    models = []
    for model_name in model_names:
        ckpt_path = os.path.join(checkpoint_dir, model_name, "best_model.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model_args = filter_model_args(checkpoint["params"])
        encoder = build_encoder()
        model = EncoderDecoder(resnet_encoder=encoder, **model_args)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()
        models.append(model)
    n_models = len(models)
    n_imgs = len(indices)
    fig, axs = plt.subplots(n_imgs, 2 + n_models, figsize=(4*(2+n_models), 4*n_imgs))
    for row, idx in enumerate(indices):
        img, gt = ds_test[idx]
        axs[row,0].imshow(img.permute(1,2,0).cpu().numpy())
        axs[row,0].set_title(f"Original [{idx}]")
        axs[row,0].axis('off')
        axs[row,1].imshow(gt.cpu().numpy(), cmap='gray')
        axs[row,1].set_title("Ground Truth")
        axs[row,1].axis('off')
        for col, model in enumerate(models):
            with torch.no_grad():
                pred = model(img.unsqueeze(0).to(device))
                pred_mask = pred.argmax(dim=1).cpu().squeeze().numpy()
            axs[row,2+col].imshow(pred_mask, cmap='gray')
            axs[row,2+col].set_title(f"{model_names[col]}")
            axs[row,2+col].axis('off')
    plt.tight_layout()
    plt.show()


# Exemplo de uso:
# df_perf = evaluate_models_performance("./checkpoints", "./data/oxford_pets", ["strides_32", "strides_2", ...])
# plot_model_segmentations("./checkpoints", "./data/oxford_pets", ["strides_32", ...], indices=[0,5,10])
# df_cost = compute_model_cost("./checkpoints", ["strides_32", ...])
