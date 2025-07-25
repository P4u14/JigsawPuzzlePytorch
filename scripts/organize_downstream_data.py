import os
import shutil
import random
import pandas as pd
from pathlib import Path

def main():
    # Define the source and destination directories using Path for OS compatibility
    source_dir = Path.home() / "Documents" / "DA" / "AtlasDaten-BMI-Percentile"
    dest_dir = Path.home() / "Documents" / "DA" / "Downstream"

    print(f"Quellordner: {source_dir}")
    print(f"Zielordner: {dest_dir}")

    if not source_dir.exists():
        print(f"FEHLER: Der Quellordner '{source_dir}' wurde nicht gefunden.")
        return

    # Create destination directories if they don't exist
    train_dir = dest_dir / "ILSVRC2012_img_train"
    val_dir = dest_dir / "ILSVRC2012_img_val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Collect all image pairs ---
    image_pairs = []
    print("\nSuche nach Bildern...")
    # Use rglob to recursively find all .Gauss.png files, excluding masks
    all_gauss_files = list(source_dir.rglob("*.Gauss.png"))
    print(f"{len(all_gauss_files)} '*.Gauss.png' Dateien gefunden.")

    for img_path in all_gauss_files:
        # Wir verarbeiten nur Bilddateien, keine Masken-Dateien in diesem Loop
        if "-mask" in img_path.name:
            continue

        print(f"\nVerarbeite Bild: {img_path.name}")
        base_name = img_path.name.replace('.Gauss.png', '')
        mask_name = f"{base_name}-mask.Gauss.png"
        mask_path = img_path.with_name(mask_name)
        print(f"  Erwartete Maske: {mask_path.name}")

        if mask_path.exists():
            print("  -> Maske gefunden.")
            image_pairs.append((img_path, mask_path))
        else:
            print("  -> FEHLER: Maske nicht gefunden.")

    print(f"\n{len(image_pairs)} vollständige Bildpaare (Bild + Maske) gefunden.")

    if not image_pairs:
        print("Keine Bildpaare zum Verschieben gefunden. Skript wird beendet.")
        return

    # --- 2. Shuffle and split the data ---
    random.shuffle(image_pairs)
    train_pairs = image_pairs[:100]
    val_pairs = image_pairs[100:]

    print(f"Aufteilung: {len(train_pairs)} Paare für 'train', {len(val_pairs)} Paare für 'val'.")

    # --- 3. & 4. Copy files and prepare CSV data ---
    csv_data = []

    def process_files(pairs, destination_folder, usage):
        filenames = []
        for img_path, mask_path in pairs:
            # Copy files
            shutil.copy(str(img_path), str(destination_folder))
            shutil.copy(str(mask_path), str(destination_folder))
            filenames.append(img_path.name)
            filenames.append(mask_path.name)

            # Extract information for CSV from the original path
            parts = img_path.relative_to(source_dir).parts
            if len(parts) >= 4:
                dataset = parts[0]
                diers_pat_id = parts[1]
                # Get DIERS_Mess-ID and remove content from '{' onwards
                diers_mess_id = parts[2].split('{')[0]
                bild_id = img_path.stem # The filename without extension

                csv_data.append({
                    "Dataset": dataset,
                    "DIERS_Pat-ID": diers_pat_id,
                    "DIERS_Mess-ID": diers_mess_id,
                    "BildID": bild_id,
                    "Usage": usage
                })
        return filenames

    train_filenames = process_files(train_pairs, train_dir, "train")
    val_filenames = process_files(val_pairs, val_dir, "val")

    # --- 5. Create and save the CSV file ---
    df = pd.DataFrame(csv_data)
    csv_path = dest_dir / "downstream_images.csv"
    df.to_csv(csv_path, index=False)

    # --- 6. Create ILSVRC-style text files ---
    def write_ilsvrc_file(filepath, filenames):
        with open(filepath, 'w') as f:
            for name in sorted(filenames): # Sortieren für konsistente Ausgabe
                f.write(f"{name} 0\n")

    train_txt_path = dest_dir / "ilsvrc12_train.txt"
    val_txt_path = dest_dir / "ilsvrc12_val.txt"
    write_ilsvrc_file(train_txt_path, train_filenames)
    write_ilsvrc_file(val_txt_path, val_filenames)


    print(f"Verarbeitung abgeschlossen.")
    print(f"{len(train_pairs)} Bildpaare nach '{train_dir}' kopiert.")
    print(f"{len(val_pairs)} Bildpaare nach '{val_dir}' kopiert.")
    print(f"CSV-Datei erstellt unter: '{csv_path}'")
    print(f"ILSVRC-Train-Datei erstellt unter: '{train_txt_path}'")
    print(f"ILSVRC-Val-Datei erstellt unter: '{val_txt_path}'")

if __name__ == "__main__":
    main()
