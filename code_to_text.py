import os
import json
import nbformat

# --- Configuration ---
# Dossiers et fichiers système à ignorer dans l'arborescence et la lecture
IGNORE_LIST = {
    '.git', '__pycache__', 'venv', '.venv', 'node_modules', 
    '.ipynb_checkpoints', '.DS_Store', 'contexte_projet.txt', '.idea', '.vscode'
}

# Extension dont on veut extraire le contenu (Code)
CONTENT_EXTENSIONS = {'.ipynb'} 

OUTPUT_FILE = "contexte_projet_notebooks.txt"

def extract_notebook_content(filepath):
    """
    Lit un fichier .ipynb avec nbformat et extrait uniquement 
    les cellules de type CODE et MARKDOWN.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
            content = []
            for i, cell in enumerate(nb.cells):
                cell_type = cell.cell_type.upper()
                source = cell.source
                # On ajoute une séparation claire entre les cellules
                content.append(f"--- Cellule {i} [{cell_type}] ---\n{source}\n")
            return "\n".join(content)
    except Exception as e:
        return f"[Erreur lors de la lecture du Notebook : {e}]\n"

def generate_context_file(root_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as f_out:
        
        # --- 1. GÉNÉRATION DE L'ARBORESCENCE COMPLÈTE ---
        f_out.write("### STRUCTURE COMPLÈTE DU PROJET ###\n")
        
        for root, dirs, files in os.walk(root_dir):
            # Filtrage des dossiers interdits (in-place)
            dirs[:] = [d for d in dirs if d not in IGNORE_LIST]
            
            # Calcul de l'indentation
            level = root.replace(root_dir, '').count(os.sep)
            indent = ' ' * 4 * level
            
            # Écriture du nom du dossier
            f_out.write(f"{indent}{os.path.basename(root)}/\n")
            
            # Écriture de TOUS les fichiers du dossier (sauf ignorés)
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                if f not in IGNORE_LIST:
                    f_out.write(f"{sub_indent}{f}\n")
        
        f_out.write("\n" + "="*50 + "\n\n")

        # --- 2. EXTRACTION DU CONTENU DES NOTEBOOKS UNIQUEMENT ---
        for root, dirs, files in os.walk(root_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_LIST]
            
            for file in files:
                # On ne prend QUE les .ipynb pour le contenu
                if file.endswith('.ipynb'):
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, root_dir)
                    
                    f_out.write(f"#### CONTENU DU FICHIER : {relative_path} ####\n")
                    f_out.write(extract_notebook_content(full_path))
                    f_out.write(f"\n#### FIN DU FICHIER : {relative_path} ####\n\n")

if __name__ == "__main__":
    chemin_actuel = os.getcwd()
    print(f"Analyse du projet dans : {chemin_actuel}")
    print(f"1. Génération de l'arborescence complète.")
    print(f"2. Extraction du code des fichiers .ipynb uniquement.")
    
    generate_context_file(chemin_actuel, OUTPUT_FILE)
    
    print(f"Terminé ! Le fichier '{OUTPUT_FILE}' a été généré.")