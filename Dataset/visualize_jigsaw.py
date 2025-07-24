import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import random

# Bildpfad und Permutationsdatei anpassen
image_path = "demo_images_small/ILSVRC2012_img_val/634913656173116806.Gauss.png"  # Beispielbild
perm_file = "permutations_1000.npy"

# Parameter
patch_size = 75
grid_size = 3

# Transformation
transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor()
])

# Lade Bild
img = Image.open(image_path).convert("RGB")
img_tensor = transform(img)

# Extrahiere 9 Patches
patches = []
for i in range(grid_size):
    for j in range(grid_size):
        top = i * patch_size
        left = j * patch_size
        patch = img_tensor[:, top:top+patch_size, left:left+patch_size]
        patches.append(patch)

# Wähle zufällige Permutation
perms = np.load(perm_file)
perm_index = random.randint(0, len(perms)-1)
perm = perms[perm_index]

print(f"Permutation ID: {perm_index}")
print(f"Permutation: {perm}")

# Reordne Patches gemäß Permutation
shuffled = [patches[i] for i in perm]

# Visualisierung
fig, axs = plt.subplots(3, 3, figsize=(5, 5))
for idx, patch in enumerate(shuffled):
    row = idx // 3
    col = idx % 3
    axs[row, col].imshow(patch.permute(1, 2, 0))
    axs[row, col].axis("off")
plt.suptitle(f"Permutiertes Bild (ID {perm_index})")
plt.tight_layout()
plt.show()
