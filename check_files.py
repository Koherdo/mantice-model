# check_files.py
import os
import glob

def check_saved_files():
    project_root = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_root, 'results', 'figures')
    
    print(f"Recherche dans: {results_dir}")
    
    if os.path.exists(results_dir):
        pdf_files = glob.glob(os.path.join(results_dir, "*.pdf"))
        if pdf_files:
            print("Fichiers PDF trouvés:")
            for file in pdf_files:
                print(f"  - {os.path.basename(file)}")
        else:
            print("Aucun fichier PDF trouvé dans le dossier results/figures/")
    else:
        print("Le dossier results/figures/ n'existe pas")

if __name__ == "__main__":
    check_saved_files()