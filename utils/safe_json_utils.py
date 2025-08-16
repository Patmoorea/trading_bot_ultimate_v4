import json, os, shutil


def atomic_save_json(data, path):
    """Sauvegarde atomique d'un JSON (jamais corrompu)."""
    temp_path = path + ".tmp"
    with open(temp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temp_path, path)


def safe_update_shared_data(new_data, path="src/shared_data.json"):
    """Sauvegarde avec backup automatique."""
    backup_path = path + ".bak"
    if os.path.exists(path):
        shutil.copy2(path, backup_path)
    atomic_save_json(new_data, path)


def safe_load_shared_data(path="src/shared_data.json"):
    """Lecture sécurisée (récupère le backup si le JSON est corrompu)."""
    backup_path = path + ".bak"
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Lecture JSON: {e}")
        if os.path.exists(backup_path):
            print("[INFO] Restauration du backup car JSON corrompu.")
            shutil.copy2(backup_path, path)
            with open(path, "r") as f:
                return json.load(f)
        else:
            print("[WARNING] Aucun backup disponible, on repart à zéro.")
            return {}
