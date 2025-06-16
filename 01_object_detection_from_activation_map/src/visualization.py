"""
Module for visualization utilities for object detection using activation maps.
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
import torch
import matplotlib.patches as patches
from object_detection import detect_max_activation, map_activation_to_original, estimate_bounding_box_from_activation

class ObjectDetectionVisualizer:
    """
    Utility class for visualizing activation maps and detected positions on images.

    Methods:
        show_images_grid(image_dir, image_names, n_rows=2, n_cols=5):
            Display a grid of images from a directory.
        plot_activation_and_detection(img_path, model, transform, image_names, image_dir, img_id=0):
            Show activation map and detected position on the original image.
    """

    @staticmethod
    def show_images_grid(image_dir, image_names, n_rows=2, n_cols=5):
        """
        Display a grid of images from a directory, sorted alphabetically.

        Args:
            image_dir (str): Path to the directory containing images.
            image_names (list): List of image filenames.
            n_rows (int): Number of rows in the grid.
            n_cols (int): Number of columns in the grid.
        """
        sorted_names = sorted(image_names)
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        for i, name in enumerate(sorted_names[:n_rows*n_cols]):
            img = Image.open(os.path.join(image_dir, name))
            axs[i//n_cols, i%n_cols].imshow(img)
            axs[i//n_cols, i%n_cols].set_title(name)
            axs[i//n_cols, i%n_cols].axis('off')
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_activation_and_detection(
        img_path, model, transform, image_names, image_dir, img_id=0, plot_bbox=False, bbox_threshold=0.5
    ):
        # ...existing code...
        name_img = image_names[img_id]
        img = Image.open(os.path.join(image_dir, name_img)).convert("RGB")
        img_resized = img.resize((224, 224))
        img_t = transform(img_resized)
        with torch.no_grad():
            activation_map = model(img_t.unsqueeze(0))
        from object_detection import detect_max_activation, map_activation_to_original, estimate_bounding_box_from_activation
        pos_max = detect_max_activation(activation_map)
        activation_size = activation_map.shape[-1]
        pos_img = map_activation_to_original(pos_max, (224, 224), resized_size=(224, 224), map_size=(activation_size, activation_size))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(activation_map.cpu(), cmap='viridis')
        ax1.set_title('Activation Map')
        ax1.scatter([pos_max[1]], [pos_max[0]], color='red', s=120, marker='x')
        ax2.imshow(img_resized)
        ax2.scatter([pos_img[0]], [pos_img[1]], color='red', s=200, marker='x')
        ax2.set_title(f'Detected position: ({int(pos_img[0])}, {int(pos_img[1])})')
        # Plotar bounding box estimada se solicitado
        if plot_bbox:
            bbox = estimate_bounding_box_from_activation(activation_map, threshold_ratio=bbox_threshold)
            # bbox: (min_col, min_row, max_col, max_row) em coordenadas do mapa de ativação
            # Converter para coordenadas da imagem 224x224
            top_left = map_activation_to_original((bbox[1], bbox[0]), (224, 224), resized_size=(224, 224), map_size=(activation_size, activation_size))
            bottom_right = map_activation_to_original((bbox[3], bbox[2]), (224, 224), resized_size=(224, 224), map_size=(activation_size, activation_size))
            width = bottom_right[0] - top_left[0]
            height = bottom_right[1] - top_left[1]
            rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor='lime', facecolor='none')
            ax2.add_patch(rect)
        plt.show()

    @staticmethod
    def plot_detection_grid(models, model_names, image_dir, image_names, transform, plot_bbox=False, bbox_threshold=0.5, save_pdf=False, pdf_path='./detection_grid.pdf'):
        """
        Display a grid with detected positions for each model and image, using images resized to 224x224.
        Each row corresponds to a model and each column to an image. The detected position is shown directly on the resized image.

        Args:
            models (list): List of models to use for detection.
            model_names (list): List of model names (str), one for each model.
            image_dir (str): Directory containing the images.
            image_names (list): List of image filenames.
            transform (callable): Transformations to apply to the images.
            plot_bbox (bool): Whether to plot the estimated bounding box.
            bbox_threshold (float): Threshold ratio for bounding box estimation.
            save_pdf (bool): If True, saves the grid as a PDF.
            pdf_path (str): Path to save the PDF file.
        """
        n_models = len(models)
        n_imgs = len(image_names)
        fig, axs = plt.subplots(n_models, n_imgs, figsize=(5*n_imgs, 5*n_models), squeeze=False)
        from object_detection import detect_max_activation, map_activation_to_original, estimate_bounding_box_from_activation
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            for j, img_name in enumerate(image_names):
                img_path = os.path.join(image_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((224, 224))
                img_t = transform(img_resized)
                with torch.no_grad():
                    activation_map = model(img_t.unsqueeze(0))
                pos_max = detect_max_activation(activation_map)
                activation_size = activation_map.shape[-1]
                pos_img = map_activation_to_original(pos_max, (224, 224), resized_size=(224, 224), map_size=(activation_size, activation_size))
                ax = axs[i, j]
                ax.imshow(img_resized)
                ax.scatter([pos_img[0]], [pos_img[1]], color='red', s=200, marker='x')
                if plot_bbox:
                    bbox = estimate_bounding_box_from_activation(activation_map, threshold_ratio=bbox_threshold)
                    top_left = map_activation_to_original((bbox[1], bbox[0]), (224, 224), resized_size=(224, 224), map_size=(activation_size, activation_size))
                    bottom_right = map_activation_to_original((bbox[3], bbox[2]), (224, 224), resized_size=(224, 224), map_size=(activation_size, activation_size))
                    width = bottom_right[0] - top_left[0]
                    height = bottom_right[1] - top_left[1]
                    rect = patches.Rectangle(top_left, width, height, linewidth=2, edgecolor='lime', facecolor='none')
                    ax.add_patch(rect)
                ax.set_title(f"Detected position: ({int(pos_img[0])}, {int(pos_img[1])})", fontsize=18)
                ax.axis('off')
        
        for i, model_name in enumerate(model_names):
            y_position = 1 - (i + 0.5) / n_models
            fig.text(0.0, y_position, model_name, fontsize=16, rotation=90, 
                    va='center', ha='left')
        plt.tight_layout()
        plt.subplots_adjust(left=0.008)  # espaço mínimo para os títulos
        if save_pdf:
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.show()
        

        
        
    @staticmethod
    def plot_activation_map_grid(models, model_names, image_dir, image_names, transform, save_pdf=False, pdf_path='./activation_maps.pdf'):
        """
        Exibe uma grade onde a primeira linha mostra as imagens 224x224 e as linhas seguintes mostram os mapas de ativação de cada modelo,
        com o máximo marcado e título indicando a posição detectada no mapa 14x14 e a posição correspondente na imagem 224x224.

        Args:
            models (list): Lista de modelos para extração dos mapas de ativação.
            model_names (list): Lista de nomes dos modelos.
            image_dir (str): Diretório das imagens.
            image_names (list): Lista de nomes das imagens.
            transform (callable): Transformação a ser aplicada nas imagens.
            save_pdf (bool): Se True, salva o PDF do grid.
            pdf_path (str): Caminho do arquivo PDF a ser salvo.
        """
        n_models = len(models)
        n_imgs = len(image_names)
        fig, axs = plt.subplots(n_models+1, n_imgs, figsize=(4*n_imgs, 4*(n_models+1)), squeeze=False)
        # Primeira linha: imagens
        for j, img_name in enumerate(image_names):
            img_path = os.path.join(image_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            img_resized = img.resize((224, 224))
            ax = axs[0, j]
            ax.imshow(img_resized)
            ax.set_title(img_name, fontsize=14)
            ax.axis('off')
        # Demais linhas: mapas de ativação
        
        for i, (model, model_name) in enumerate(zip(models, model_names)):
            for j, img_name in enumerate(image_names):
                img_path = os.path.join(image_dir, img_name)
                img = Image.open(img_path).convert("RGB")
                img_resized = img.resize((224, 224))
                img_t = transform(img_resized)
                with torch.no_grad():
                    activation_map = model(img_t.unsqueeze(0))
                pos_max = detect_max_activation(activation_map)
                activation_size = activation_map.shape[-1]
                x, y = pos_max[1], pos_max[0]
                pos_img = map_activation_to_original(pos_max, (224, 224), resized_size=(224, 224), map_size=(activation_size, activation_size))
                ax = axs[i+1, j]
                ax.imshow(activation_map.cpu(), cmap='viridis')
                ax.scatter([x], [y], color='red', s=120, marker='x')
                ax.set_title(f"Max Activation Position:({x}, {y})", fontsize=17)
                ax.axis('off')
        # Adicionar nomes dos modelos como títulos do eixo y usando fig.text
        for i, model_name in enumerate(["Image"] + model_names):
            y_position = 1 - (i + 0.5) / (n_models+1)
            fig.text(0.0, y_position, model_name, fontsize=18, rotation=90, va='center', ha='left')
        plt.tight_layout()
        plt.subplots_adjust(left=0.04)
        if save_pdf:
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
        plt.show()