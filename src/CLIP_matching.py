import torch
import clip
from PIL import Image
from DATA import Data
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def read_data_test():
    data_text = Data(f"data/RT-Dataset/rt-test-text.txt").list_dict
    data_image = Data(f"data/RT-Dataset/rt-test-img.txt").list_dict

    list_text = [data['title'] for data in data_text]
    text_tokenized = clip.tokenize(list_text, truncate=True).to(device)
    print("Len texts: ", len(list_text))
    
    list_prob = []
    dict_text_image_score = defaultdict(list)
    
    for idx, single_data_image in tqdm(enumerate(data_image)):
        # image = preprocess(Image.open(f"./data/RT-Dataset/images-Test/{single_data_image['imgFile']}")).unsqueeze(0).to(device)
        image = preprocess(Image.open(f"./data/RT-Dataset/images-Test/{single_data_image['hashvalue']}")).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text_tokenized)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        for text_idx, prob in enumerate(probs):
            dict_text_image_score[text_idx].append((single_data_image['iid'], prob))
    
    f = open("./file_submit/RT-Dataset.txt", 'w')
    for key in tqdm(dict_text_image_score.keys()):
        line_ans = ""
        
        article_id = data_text[key]['url']
        line_ans = f"{article_id}\t"
        
        idd_sorted_score = sorted(dict_text_image_score[key], key=lambda tup: tup[1], reverse=True)
        for sort_iid in idd_sorted_score[:100]:
            line_ans += f"{sort_iid[0]}\t"
        line_ans += "\n"
        f.write(line_ans)
    
    
def read_data_train():
    list_data = Data("data/GDELT-Dataset/GDELT-P1-Training.txt").list_dict[:10]

    list_text = [data['title'] for data in list_data]
    text_tokenized = clip.tokenize(list_text).to(device)
    print("Len texts: ", len(list_text))
    
    list_image = [preprocess(Image.open(f"./data/GDELT-Dataset/GDELT-P1-Training/{data['imgFile']}")).unsqueeze(0).to(device) for data in list_data]
    print("Len image: ", len(list_image))
    
    list_prob = []
    for idx, image in enumerate(list_image):
        print(list_data[idx]['imgFile'])
        with torch.no_grad():
            logits_per_image, logits_per_text = model(image, text_tokenized)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        list_prob.append(probs)
        print("Label probs:", probs)
        argmax = np.argmax(probs, axis=1)[0]
        print(list_data[argmax]['title'])

if __name__=='__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    read_data_test()
    
    
    
    