import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Deine Pfade anpassen:
image_folder = 'demo_images'              # Eingabeordner
image_list_txt = 'demo_images/demo_images.txt'        # Liste mit Bildpfaden
output_folder = 'demo_images_small'     # Zielordner

# Resize + CenterCrop
transform = transforms.Compose([
    transforms.Resize(256, interpolation=Image.BILINEAR),
    transforms.CenterCrop(255)
])

# Erstelle Zielordner
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Lese Bildliste aus Textdatei
with open(image_list_txt, 'r') as f:
    lines = f.readlines()

# Verarbeite alle Bilder
for line in tqdm(lines):
    fname = line.strip().split()[0]
    input_path = os.path.join(image_folder, fname)
    output_path = os.path.join(output_folder, fname)

    if os.path.exists(output_path):
        continue

    try:
        img = Image.open(input_path).convert('RGB')
        img = transform(img)
        img.save(output_path)
    except Exception as e:
        print(f"Fehler bei {fname}: {e}")
