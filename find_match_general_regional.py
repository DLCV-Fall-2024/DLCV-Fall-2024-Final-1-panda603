from datasets import load_dataset
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
import json
from tqdm import tqdm

def load_and_process_datasets():
    dataset = load_dataset("ntudlcv/dlcv_2024_final1", split="train")
    
    general_data = []
    regional_data = []
    
    print("Processing and sorting images...")
    for item in tqdm(dataset, desc="Processing dataset"):
        task_type = item['id'].split('_')[1]
        idx = int(item['id'].split('_')[2])
        
        if task_type == 'general':
            general_data.append((idx, item['id'], np.array(item['image'])))
        elif task_type == 'regional':
            regional_data.append((idx, item['id'], np.array(item['image'])))
    
    general_data.sort(key=lambda x: x[0])
    regional_data.sort(key=lambda x: x[0])
    
    return general_data, regional_data

def find_sequential_matches(general_data, regional_data, threshold=0.95):
    matching_pairs = {}
    regional_idx = 0
    
    for gen_idx, gen_id, gen_img in tqdm(general_data, desc="Finding matches"):
        gen_gray = cv2.cvtColor(gen_img, cv2.COLOR_RGB2GRAY)
        matches = []
        
        while regional_idx < len(regional_data):
            reg_idx, reg_id, reg_img = regional_data[regional_idx]
            reg_gray = cv2.cvtColor(reg_img, cv2.COLOR_RGB2GRAY)
            
            if reg_gray.shape != gen_gray.shape:
                reg_gray = cv2.resize(reg_gray, (gen_gray.shape[1], gen_gray.shape[0]))
            
            score = ssim(gen_gray, reg_gray)
            print(f"Comparing {gen_id} with {reg_id}, score: {score}")
            
            if score > threshold:
                matches.append(reg_id)
                regional_idx += 1
            else:
                break
        
        if matches:
            matching_pairs[gen_id] = matches
        with open('task_matching_pairs.json', 'w') as f:
            json.dump(matching_pairs, f, indent=4)
    
    return matching_pairs

def main():
    print("Loading datasets...")
    general_data, regional_data = load_and_process_datasets()
    
    print("Finding sequential matches...")
    matching_pairs = find_sequential_matches(general_data, regional_data)
    
    # print("Saving results...")
    # with open('task_matching_pairs.json', 'w') as f:
    #     json.dump(matching_pairs, f, indent=4)
    
    print("\nMatching Statistics:")
    for gen_id, matches in matching_pairs.items():
        print(f"{gen_id}: {len(matches)} matches")
        print(f"Matches: {matches}")
        print("-" * 50)

if __name__ == "__main__":
    main()