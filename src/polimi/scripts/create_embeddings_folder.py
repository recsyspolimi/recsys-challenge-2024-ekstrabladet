import os
import shutil

def create_folder_with_files(base_path, folders, output_folder_name):
    if not os.path.exists(base_path):
        print(f"Error: Path {base_path} does not exist.")
        return
    
    output_path = os.path.join(base_path, output_folder_name)
    os.makedirs(output_path, exist_ok=True)
    
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder} does not exist in {base_path}. Skipping.")
            continue
        
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                shutil.copy2(file_path, output_path)
                print(f"Copied file {file} from {folder} to {output_folder_name}")
    
    print(f"Copying operation completed. Files from folders {folders} have been copied to {output_folder_name}.")
    
    # Remove .DS_Store files in the output folder if they exist
    remove_ds_store_files(output_path)

def remove_ds_store_files(folder_path):
    ds_store_files = [f for f in os.listdir(folder_path) if f == '.DS_Store']
    for ds_store_file in ds_store_files:
        os.remove(os.path.join(folder_path, ds_store_file))
        print(f"Removed .DS_Store file: {ds_store_file}")

if __name__ == "__main__":
    base_path = "~/dataset/"
    folders_to_copy = ["Ekstra_Bladet_contrastive_vector", "Ekstra_Bladet_word2vec", "FacebookAI_xlm_roberta_base", "google_bert_base_multilingual_cased"]
    output_folder_name = "embeddings"
    
    create_folder_with_files(base_path, folders_to_copy, output_folder_name)
