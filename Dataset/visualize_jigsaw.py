import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random
import torch

# Bildpfad und Permutationsdatei anpassen
image_path = "demo_images_small/ILSVRC2012_img_val/634913656173116806.Gauss.png"  # Beispielbild
perm_file = "permutations_1000.npy"

# Parameter
patch_size = 75
grid_size = 3
gap = 5  # Lücke zwischen den Patches

# Transformation
resize_size = grid_size * patch_size + (grid_size - 1) * gap
transform = transforms.Compose([
    transforms.Resize((resize_size, resize_size)),
    transforms.ToTensor()
])

# Lade Bild
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img)

# Extrahiere 9 Patches
patches = []
for i in range(grid_size):
    for j in range(grid_size):
        top = i * (patch_size + gap)
        left = j * (patch_size + gap)
        patch = img_tensor[:, top:top+patch_size, left:left+patch_size]
        patches.append(patch)

# Wähle zufällige Permutation
perms = np.load(perm_file)
perm_index = random.randint(0, len(perms)-1)
perm = perms[perm_index]

# Permutierte Patches anordnen
permuted_patches = [patches[i] for i in perm]

# Erstelle eine leere Leinwand für das permutierte Puzzle
canvas_size = grid_size * patch_size + (grid_size - 1) * gap
permuted_canvas = torch.ones((3, canvas_size, canvas_size))  # Weißer Hintergrund

# Platziere die permutierten Patches auf der Leinwand
for i, patch in enumerate(permuted_patches):
    row = i // grid_size
    col = i % grid_size
    top = row * (patch_size + gap)
    left = col * (patch_size + gap)
    permuted_canvas[:, top:top+patch_size, left:left+patch_size] = patch

# Erstelle eine leere Leinwand für das gelöste Puzzle
solved_canvas = torch.ones((3, canvas_size, canvas_size))  # Weißer Hintergrund

# Platziere die sortierten Patches auf der Leinwand
for i, patch in enumerate(patches):
    row = i // grid_size
    col = i % grid_size
    top = row * (patch_size + gap)
    left = col * (patch_size + gap)
    solved_canvas[:, top:top+patch_size, left:left+patch_size] = patch


# Konvertiere zu PIL-Bildern für die Anzeige
permuted_img = transforms.ToPILImage()(permuted_canvas)
solved_img = transforms.ToPILImage()(solved_canvas)

# Zeige die Bilder an
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(permuted_img)
axes[0].set_title("Permuted Puzzle")
axes[0].axis('off')

axes[1].imshow(solved_img)
axes[1].set_title("Solved Puzzle")
axes[1].axis('off')

plt.show()
