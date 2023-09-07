import os
import numpy as np
import h5py
import json
import torch
# from scipy.misc import imread, imresize
from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample, shuffle

def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image, min_word_freq, output_folder,
                       max_len=50, max_tag_len=20):
    """
    Creates input files for training, validation, and test data.

    :param dataset: name of dataset, one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path: path of Karpathy JSON file with splits and captions
    :param image_folder: folder with downloaded images
    :param captions_per_image: number of captions to sample per image
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s
    :param output_folder: folder to save files
    :param max_len: don't sample captions longer than this length
    """

    # assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    image_paths = []
    image_captions = []
    image_tags_yn=[]
    image_tags=[]
    image_l_paths = []
    image_l_captions = []
    image_l_tags_yn=[]
    image_l_tags=[]
    # val_image_paths = []
    # val_image_captions = []
    # val_image_tags_yn=[]
    # val_image_tags=[]
    test_image_paths = []
    test_image_captions = []
    test_image_tags_yn=[]
    test_image_tags=[]
    test_image_l_paths = []
    test_image_l_captions = []
    test_image_l_tags_yn=[]
    test_image_l_tags=[]
    # word_freq = Counter()
    # tag_freq=Counter()
    word_map_file = os.path.join(output_folder, 'WORDMAP_' + dataset + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)  #1198, dict{word:No.}

    tag_map_file=os.path.join(output_folder,'TAGMAP_'+dataset+'.json')
    with open(tag_map_file,'r') as j:
        tag_map=json.load(j)
    tags_list = list(tag_map.keys())[:-4]
    # tags_list=["normal", "calcified", "granuloma", "lung", "upper", "lobe", "right", "opacity", "pulmonary", "atelectasis", "lingula", "markings", "bilateral", "interstitial", "prominent", "left", "density", "calcinosis", "base", "bone", "spine", "aorta", "tortuous", "degenerative", "catheters", "indwelling", "thoracic", "vertebrae", "mild", "cardiomegaly", "technical", "quality", "of", "image", "unsatisfactory", "hypoinflation", "diaphragm", "elevated", "congestion", "chronic", "severe", "costophrenic", "angle", "blunted", "surgical" "instruments", "airspace", "disease", "pleural", "effusion", "medical", "device", "patchy", "streaky", "pleura", "thickening", "hilum", "round", "lower", "cicatrix", "focal", "small", "hyperdistention", "mediastinum", "nodule", "no", "indexing", "posterior", "scoliosis", "bronchovascular", "granulomatous", "osteophyte", "multiple", "middle", "emphysema", "atherosclerosis", "lymph", "nodes", "deformity", "anterior", "ribs", "scattered", "lumbar", "flattened", "spondylosis", "thorax", "fractures", "infiltrate", "moderate", "cardiac", "shadow", "enlarged", "apex", "abdomen", "large"]
    # tags_list=["normal", "calcified", "granuloma", "lung", "upper", "lobe", "right", "opacity", "pulmonary", "atelectasis": 10, "lingula": 11, "markings": 12, "bilateral": 13, "interstitial": 14, "prominent": 15, "left": 16, "density": 17, "calcinosis": 18, "base": 19, "bone": 20, "spine": 21, "aorta": 22, "tortuous": 23, "degenerative": 24, "catheters": 25, "indwelling": 26, "thoracic": 27, "vertebrae": 28, "mild": 29, "cardiomegaly": 30, "technical": 31, "quality": 32, "of": 33, "image": 34, "unsatisfactory": 35, "hypoinflation": 36, "diaphragm": 37, "elevated": 38, "congestion": 39, "chronic": 40, "severe": 41, "costophrenic": 42, "angle": 43, "blunted": 44, "surgical": 45, "instruments": 46, "airspace": 47, "disease": 48, "pleural": 49, "effusion": 50, "medical": 51, "device": 52, "patchy": 53, "streaky": 54, "pleura": 55, "thickening": 56, "hilum": 57, "round": 58, "lower": 59, "cicatrix": 60, "focal": 61, "small": 62, "hyperdistention": 63, "mediastinum": 64, "nodule": 65, "no": 66, "indexing": 67, "posterior": 68, "scoliosis": 69, "bronchovascular": 70, "granulomatous": 71, "osteophyte": 72, "multiple": 73, "middle": 74, "emphysema": 75, "atherosclerosis": 76, "lymph": 77, "nodes": 78, "deformity": 79, "anterior": 80, "ribs": 81, "scattered": 82, "lumbar": 83, "flattened": 84, "spondylosis": 85, "thorax": 86, "fractures": 87, "infiltrate": 88, "moderate": 89, "cardiac": 90, "shadow": 91, "enlarged": 92, "apex": 93, "abdomen": 94, "large": 95]
    # print(len(tags_list))
    # tags_list=["abdomen", "gross", "metastatic", "carcinoma", "peritoneum", "tuberculosis", "abscess", "acute", "prostate", "in", "necrosis", "leukemia", "mesothelioma", "fibrinous", "body", "burn", "polycystic","of", "disease", "teratoma", "newborn", "cord", "with", "syndrome", "breast", "adenocarcinoma", "cardiovascular", "heart", "vasculature", "aorta", "prosthetic", "valve", "aortic", "mitral", "hypertrophy", "left", "and", "right", "pericarditis", "infarct", "remote", "bacterial", "endocarditis", "thrombotic", "coronary", "artery", "thrombosis", "chronic", "ischemia", "normal", "stenosis", "scar", "rheumatic", "atherosclerosis", "plaque", "graft", "pulmonary", "nonbacterial", "saphenous", "vein", "bypass", "origin", "healed", "ruptured", "atherosclerotic", "hemorrhage", "thrombus", "ventricle", "berry", "aneurysm", "circle", "willis", "nervous", "system", "abdominal", "dissecting", "myocardial", "type", "tricuspid", "valvulitis", "infected", "septal", "dysplasia", "secondary", "to", "lung", "fibrosis", "shock", "malignant", "anomalous", "ventricular", "dilated", "cardiomyopathy", "cerebral", "defect", "atrial", "traumatic", "sinus", "mural", "prosthesis", "due", "hemorrhagic", "candida", "abscesses", "papillary", "muscle", "rupture", "renal", "copd", "emphysema", "embolic", "infarcts", "multiple", "tumor", "aspergillus", "basilar", "lymphoma", "node", "coarctation", "melanoma", "duodenum", "petechiae", "malformation", "cell", "endocrine", "adrenal", "cortical", "adenoma", "pituitary", "hematoma", "atrophy", "meningioma", "extremities", "gangrene", "hand", "skin", "edema", "ulcer", "female", "reproductive", "herpes", "polyp", "uterus", "ovary", "cystic", "cysts", "gastrointestinal", "intestine", "esophagus", "small", "stomach", "colon", "necrotizing", "peptic", "ulcerative", "colitis", "ulcers", "varices", "hepatitis", "diffuse", "infection", "pancreas", "tissue", "purulent", "head", "bilateral", "face", "bone", "calvarium", "myeloma", "skull", "fracture", "hematologic", "spleen", "cirrhosis", "biliary", "hepatobiliary", "liver", "gallbladder", "micronodular", "endstage", "hepatoma", "kidney", "sarcoma", "brain", "spinal", "meningitis", "intraventricular", "midbrain", "hemorrhages", "glioblastoma", "multiforme", "encephalopathy", "ependymoma", "lymphatic", "lymph", "male", "interstitial", "musculoskeletal", "soft", "bones", "joints", "vertebra", "thorax", "respiratory", "trachea", "pneumonia", "bronchopneumonia", "thromboembolus", "oral", "larynx", "patient", "urinary", "bladder", "hydronephrosis", "pyelonephritis"]                                                                                                                                                                                                                                                                   
    # tags_list=["normal", "calcified", "granuloma", "lung", "upper", "lobe", "right", "opacity", "pulmonary", "atelectasis", "lingula", "markings", "bilateral", "interstitial", "diffuse", "prominent", "fibrosis", "mastectomy", "left", "density", "retrocardiac", "calcinosis", "blood", "vessels", "base", "bone", "diseases", "metabolic", "spine", "aorta", "tortuous", "shoulder", "degenerative", "catheters", "indwelling", "thoracic", "vertebrae", "mild", "cardiomegaly", "technical", "quality", "of", "image", "unsatisfactory", "hypoinflation", "diaphragm", "elevated", "congestion", "dislocations", "chronic", "severe", "consolidation", "costophrenic", "angle", "blunted", "surgical", "instruments", "airspace", "disease", "pleural", "effusion", "implanted", "medical", "device", "humerus", "patchy", "streaky", "pleura", "thickening", "hilum", "round", "lower", "cicatrix", "focal", "small", "hyperdistention", "sternum", "pneumothorax", "shift", "mediastinum", "nodule", "no", "indexing", "sulcus", "posterior", "obscured", "scoliosis", "bronchovascular", "granulomatous", "osteophyte", "multiple", "middle", "hernia", "hiatal", "emphysema", "atherosclerosis", "lymph", "nodes", "deformity", "anterior", "lucency", "ribs", "scattered", "lumbar", "flattened", "spondylosis", "clavicle", "irregular", "thorax", "fractures", "healed", "borderline", "kyphosis", "obstructive", "infiltrate", "heart", "failure", "edema", "moderate", "cardiac", "shadow", "enlarged", "breast", "foreign", "bodies", "spinal", "fusion", "cervical", "apex", "diaphragmatic", "eventration", "arthritis", "pneumonia", "cysts", "tuberculosis", "abdomen", "stents", "coronary", "hypertension", "hyperlucent", "hydropneumothorax", "large", "tube", "inserted", "sarcoidosis", "colonic", "interposition", "implants", "pneumoperitoneum", "sclerosis", "cholelithiasis", "epicardial", "fat", "and", "bones", "mass", "paratracheal", "artery", "supracardiac", "trachea", "carina", "hyperostosis", "idiopathic", "skeletal", "expansile", "lesions", "sinus", "ventricles", "pneumonectomy", "alveoli", "volume", "loss", "pericardial", "bronchi", "bronchitis", "reticular", "nipple", "adipose", "tissue", "subcutaneous", "neck", "blister", "azygos", "contrast", "media", "funnel", "chest", "abnormal", "aortic", "valve", "hypovolemia", "bronchiectasis", "cystic", "atria", "sutures", "acute", "aneurysm", "bullous", "cavitation", "hemopneumothorax", "mitral", "esophagus", "pectus", "carinatum", "bronchiolitis", "multilobar", "hemothorax", "cardiophrenic", "osteoporosis"]
    # tags_list = {"normal": 1, "calcified": 2, "granuloma": 3, "lung": 4, "upper": 5, "lobe": 6, "right": 7, "opacity": 8, "pulmonary": 9, "atelectasis": 10, "lingula": 11, "markings": 12, "bilateral": 13, "interstitial": 14, "diffuse": 15, "prominent": 16, "fibrosis": 17, "mastectomy": 18, "left": 19, "density": 20, "retrocardiac": 21, "calcinosis": 22, "blood": 23, "vessels": 24, "base": 25, "bone": 26, "diseases": 27, "metabolic": 28, "spine": 29, "aorta": 30, "tortuous": 31, "shoulder": 32, "degenerative": 33, "catheters": 34, "indwelling": 35, "thoracic": 36, "vertebrae": 37, "mild": 38, "cardiomegaly": 39, "technical": 40, "quality": 41, "of": 42, "image": 43, "unsatisfactory": 44, "hypoinflation": 45, "diaphragm": 46, "elevated": 47, "congestion": 48, "dislocations": 49, "chronic": 50, "severe": 51, "consolidation": 52, "costophrenic": 53, "angle": 54, "blunted": 55, "surgical": 56, "instruments": 57, "airspace": 58, "disease": 59, "pleural": 60, "effusion": 61, "implanted": 62, "medical": 63, "device": 64, "humerus": 65, "patchy": 66, "streaky": 67, "pleura": 68, "thickening": 69, "hilum": 70, "round": 71, "lower": 72, "cicatrix": 73, "focal": 74, "small": 75, "hyperdistention": 76, "sternum": 77, "pneumothorax": 78, "shift": 79, "mediastinum": 80, "nodule": 81, "no": 82, "indexing": 83, "sulcus": 84, "posterior": 85, "obscured": 86, "scoliosis": 87, "bronchovascular": 88, "granulomatous": 89, "osteophyte": 90, "multiple": 91, "middle": 92, "hernia": 93, "hiatal": 94, "emphysema": 95, "atherosclerosis": 96, "lymph": 97, "nodes": 98, "deformity": 99, "anterior": 100, "lucency": 101, "ribs": 102, "scattered": 103, "lumbar": 104, "flattened": 105, "spondylosis": 106, "clavicle": 107, "irregular": 108, "thorax": 109, "fractures": 110, "healed": 111, "borderline": 112, "kyphosis": 113, "obstructive": 114, "infiltrate": 115, "heart": 116, "failure": 117, "edema": 118, "moderate": 119, "cardiac": 120, "shadow": 121, "enlarged": 122, "breast": 123, "foreign": 124, "bodies": 125, "spinal": 126, "fusion": 127, "cervical": 128, "apex": 129, "diaphragmatic": 130, "eventration": 131, "arthritis": 132, "pneumonia": 133, "cysts": 134, "tuberculosis": 135, "abdomen": 136, "stents": 137, "coronary": 138, "hypertension": 139, "hyperlucent": 140, "hydropneumothorax": 141, "large": 142, "tube": 143, "inserted": 144, "sarcoidosis": 145, "colonic": 146, "interposition": 147, "implants": 148, "pneumoperitoneum": 149, "sclerosis": 150, "cholelithiasis": 151, "epicardial": 152, "fat": 153, "and": 154, "bones": 155, "mass": 156, "paratracheal": 157, "artery": 158, "supracardiac": 159, "trachea": 160, "carina": 161, "hyperostosis": 162, "idiopathic": 163, "skeletal": 164, "expansile": 165, "lesions": 166, "sinus": 167, "ventricles": 168, "pneumonectomy": 169, "alveoli": 170, "volume": 171, "loss": 172, "pericardial": 173, "bronchi": 174, "bronchitis": 175, "reticular": 176, "nipple": 177, "adipose": 178, "tissue": 179, "subcutaneous": 180, "neck": 181, "blister": 182, "azygos": 183, "contrast": 184, "media": 185, "funnel": 186, "chest": 187, "abnormal": 188, "aortic": 189, "valve": 190, "hypovolemia": 191, "bronchiectasis": 192, "cystic": 193, "atria": 194, "sutures": 195, "acute": 196, "aneurysm": 197, "bullous": 198, "cavitation": 199, "hemopneumothorax": 200, "mitral": 201, "esophagus": 202, "pectus": 203, "carinatum": 204, "bronchiolitis": 205, "multilobar": 206, "hemothorax": 207, "cardiophrenic": 208, "osteoporosis": 209}
    print("data['images'] len:", len(data['images']))
    print("data['images'] type:", type(data['images']))
    # tag_len = {'<10':0, '10-20':0, '20-30':0, '30-40':0, '40-50':0, '50-100':0, '>100':0}
    # cap_len = {'<10':0, '10-20':0, '20-30':0, '30-40':0, '40-50':0, '50-100':0, '>100':0}
    seed(123)
    shuffle(data['images'])
    print("first two data after shuffle:", data['images'][0]['image_id'], data['images'][1]['image_id'])

    def create_fold(impaths, imcaps, imtags, img_tags_yn, fold_id, split):
        # print("imtags:", imtags)
        with h5py.File(os.path.join(output_folder, str(fold_id) + '_' + split + '_IMAGES_' + dataset + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 file to store images
            images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

            print("\nReading fold %d images and captions, storing to file...\n" % fold_id)

            captions_list = []
            caplens = []

            tags_list=[]
            taglens=[]

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)
                
                # print("imtags[i]:", imtags[i]) # [[tags_i]]
                tags=sample(imtags[i],k=1) # [[tags_i]]
                # print("tags:", tags)
                # Sanity check
                assert len(captions) == captions_per_image

                # print("captions type:", type(captions))

                # Read images
                img = imread(impaths[i])
                if len(img.shape) == 2:
                    img = img[:, :, np.newaxis]
                    img = np.concatenate([img, img, img], axis=2)
                img = imresize(img, (224, 224))
                img = img.transpose(2, 0, 1)
                assert img.shape == (3, 224, 224)
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img

                for j, c in enumerate(captions):
                    # Encode captions
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

                    # Find caption lengths
                    c_len = len(c) + 2
                    captions_list.append(enc_c)
                    caplens.append(c_len)
                for j,c in enumerate(tags):
                    if len(c) > max_tag_len:
                        c = c[:max_tag_len]
                    enc_c=[tag_map['<start>']] + [tag_map.get(word, tag_map['<unk>']) for word in c] + [
                        tag_map['<end>']] + [tag_map['<pad>']] * (max_tag_len - len(c))
                    c_len=len(c)+2
                    tags_list.append(enc_c)
                    taglens.append(c_len)
            # Sanity check
            # assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
            print("images len:", len(images))
            # input("Please press the Enter key to proceed")

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(output_folder, str(fold_id) + '_' + split + '_CAPTIONS_' + dataset + '.json'), 'w') as j:
                json.dump(captions_list, j)

            with open(os.path.join(output_folder, str(fold_id) + '_' + split + '_CAPLENS_' + dataset + '.json'), 'w') as j:
                json.dump(caplens, j)

            with open(os.path.join(output_folder, str(fold_id) + '_' + split + '_TAGS_' + dataset + '.json'), 'w') as j:
                json.dump(tags_list, j)

            with open(os.path.join(output_folder, str(fold_id) + '_' + split + '_TAGLENS_' + dataset + '.json'), 'w') as j:
                json.dump(taglens, j)

            with open(os.path.join(output_folder, str(fold_id) + '_' + split + '_TAGSYN_' + dataset + '.json'), 'w') as j:
                json.dump(img_tags_yn, j)

    
    cnt = 0
    fold_id = 0
    for img in data['images']:
        if cnt >= 500 and fold_id < 10: # 还有一个原先的TEST集
            # create_fold(image_paths, image_captions, image_tags, image_tags_yn, fold_id, 'TRAIN') # fold_id从0开始
            # if len(image_l_paths):
            #     create_fold(image_l_paths, image_l_captions, image_l_tags, image_l_tags_yn, fold_id, 'REST') # fold_id从0开始
            # print("fold", fold_id, "len:", len(image_paths))
            fold_id += 1
            cnt = 0
            image_paths.clear()
            image_captions.clear()
            image_tags.clear()
            image_tags_yn.clear()
            image_l_paths.clear()
            image_l_captions.clear()
            image_l_tags.clear()
            image_l_tags_yn.clear()

        captions = []
        tags=[]
        tags.append(img['tag']) # tags: [[img1_tags]]
        # word_freq.update(img['caption']) # 词频统计 word_freq: {word1: freq1, word2: freq1, ...}
        # tag_freq.update(img['tag'])
        tags_yn=[]
        # print("tags")
        # print(tags)
        for value in tags_list:
            if value in tags[0]:
                tags_yn.append(1)
            else:
                tags_yn.append(0)
        # print("tagsyn")
        # print(tags_yn)
        #for c in img['sentences']:
            # Update word frequency
            #word_freq.update(c['tokens'])
        captions.append(img['caption'])
        # captions.append(img['caption'])
        # if len(captions) == 0:
        #     continue
        # if len(tags)==0:
        #     continue
        # if len(img['tag'])<=10:
        #     tag_len["<10"] += 1
        # if 10<len(img['tag'])<=20:
        #     tag_len["10-20"] += 1
        # if 20<len(img['tag'])<=30:
        #     tag_len["20-30"] += 1
        # if 30<len(img['tag'])<=30:
        #     tag_len["30-40"] += 1
        # if 40<len(img['tag'])<=50:
        #     tag_len["40-50"] += 1
        # if 50<len(img['tag'])<=100:
        #     tag_len["50-100"] += 1
        # if 100<len(img['tag']):
        #     tag_len[">100"] += 1
        
        # if len(img['caption'])<=10:
        #     cap_len["<10"] += 1
        # if 10<len(img['caption'])<=20:
        #     cap_len["10-20"] += 1
        # if 20<len(img['caption'])<=30:
        #     cap_len["20-30"] += 1
        # if 30<len(img['caption'])<=30:
        #     cap_len["30-40"] += 1
        # if 40<len(img['caption'])<=50:
        #     cap_len["40-50"] += 1
        # if 50<len(img['caption'])<=100:
        #     cap_len["50-100"] += 1
        # if 100<len(img['caption']):
        #     cap_len[">100"] += 1

        # path = os.path.join(image_folder, img['filepath'], img['filename']) if dataset == 'coco' else os.path.join(
        #     image_folder, img['filename'])
        #path=os.path.join
        #path=os.path.join(image_folder,img['image_id']+'.png')

        path=os.path.join(image_folder,img['image_id']+'.png')
        # image_paths.append(path)
        # image_captions.append(captions)
        # image_tags.append(tags)
        # image_tags_yn.append(tags_yn)

        # if img['dataset'] in {'train', 'val'}:
        cnt += 1
        if len(img['caption']) <= max_len:
            image_paths.append(path)
            image_captions.append(captions)
            image_tags.append(tags)
            image_tags_yn.append(tags_yn)
        else:
            image_l_paths.append(path)
            image_l_captions.append(captions)
            image_l_tags.append(tags)
            image_l_tags_yn.append(tags_yn)
        # elif img['dataset'] in {'test'}:
        #     if len(img['caption']) <= max_len:
        #         test_image_paths.append(path)
        #         test_image_captions.append(captions)
        #         test_image_tags.append(tags)
        #         test_image_tags_yn.append(tags_yn)
        #     else:
        #         test_image_l_paths.append(path)
        #         test_image_l_captions.append(captions)
        #         test_image_l_tags.append(tags)
        #         test_image_l_tags_yn.append(tags_yn)

    # print("max caption len:", max_len)
    # print("tag len:", tag_len)
    # print("caption len:", cap_len)
    # input("Please press enter to proceed")
    # # Sanity check
    # assert len(train_image_paths) == len(train_image_captions)
    # assert len(val_image_paths) == len(val_image_captions)
    # assert len(test_image_paths) == len(test_image_captions)
    # print("train paths len:", len(train_image_paths))
    # print("val paths len:", len(val_image_paths))
    # print("test paths len:", len(test_image_paths))
    # input("Please press the Enter key to proceed")

    # print("test_image_tags:", test_image_tags)
    # if len(test_image_paths):
    #     create_fold(test_image_paths, test_image_captions, test_image_tags, test_image_tags_yn, fold_id, 'TRAIN')
    #     if len(test_image_l_paths):
    #         create_fold(test_image_l_paths, test_image_l_captions, test_image_l_tags, test_image_l_tags_yn, fold_id, 'REST')
    #     print("fold", fold_id, "len:", len(test_image_paths))
    #     fold_id += 1
    # input("Please press enter key to proceed")
    # 最后一个fold，只作为训练集
    if len(image_paths):
        # create_fold(image_paths, image_captions, image_tags, image_tags_yn, fold_id, 'TRAIN') # fold_id从0开始
        create_fold(image_l_paths, image_l_captions, image_l_tags, image_l_tags_yn, fold_id, 'REST') # fold_id从0开始
    # # Create word map
    # words = [w for w in word_freq.keys() if word_freq[w] > min_word_freq]
    # word_map = {k: v + 1 for v, k in enumerate(words)}
    # word_map['<unk>'] = len(word_map) + 1
    # word_map['<start>'] = len(word_map) + 1
    # word_map['<end>'] = len(word_map) + 1
    # word_map['<pad>'] = 0

    # # print("tag_freq len:", len(tag_freq)) # 210
    # tags=[w for w in tag_freq.keys() if w not in [',','']]
    # tag_map={k: v+1 for v,k in enumerate(tags)}
    # tag_map['<unk>'] = len(tag_map) + 1
    # tag_map['<start>'] = len(tag_map) + 1
    # tag_map['<end>'] = len(tag_map) + 1
    # tag_map['<pad>'] = 0

    # print(len(tag_map)) #213

    # Create a base/root name for all output files
    # base_filename = dataset + '_' + str(captions_per_image) + '_cap_per_img_' + str(min_word_freq) + '_min_word_freq'

    # Save word map to a JSON
    # with open(os.path.join(output_folder, 'WORDMAP_' + dataset + '.json'), 'w') as j:
    #     json.dump(word_map, j)
    # with open(os.path.join(output_folder, 'TAGMAP_' + dataset + '.json'), 'w') as j:
    #     json.dump(tag_map, j)

    # # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    # seed(123)
    # for impaths, imcaps,imtags, img_tags_yn, split in [(train_image_paths, train_image_captions,train_image_tags,train_image_tags_yn, 'TRAIN'),
    #                                (val_image_paths, val_image_captions, val_image_tags,val_image_tags_yn,'VAL')
    #                                (test_image_paths, test_image_captions,test_image_tags,test_image_tags_yn, 'TEST')
    #                                ]:

    #     # print("imtags:", imtags)
    #     with h5py.File(os.path.join(output_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
    #         # Make a note of the number of captions we are sampling per image
    #         h.attrs['captions_per_image'] = captions_per_image

    #         # Create dataset inside HDF5 file to store images
    #         images = h.create_dataset('images', (len(impaths), 3, 224, 224), dtype='uint8')

    #         print("\nReading %s images and captions, storing to file...\n" % split)

    #         enc_captions = []
    #         caplens = []

    #         enc_tags=[]
    #         taglens=[]

    #         for i, path in enumerate(tqdm(impaths)):

    #             # Sample captions
    #             if len(imcaps[i]) < captions_per_image:
    #                 captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
    #             else:
    #                 captions = sample(imcaps[i], k=captions_per_image)
                
    #             # print("imtags[i]:", imtags[i]) # [[tags_i]]
    #             tags=sample(imtags[i],k=1) # [[tags_i]]
    #             # print("tags:", tags)
    #             # Sanity check
    #             assert len(captions) == captions_per_image

    #             # Read images
    #             img = imread(impaths[i])
    #             if len(img.shape) == 2:
    #                 img = img[:, :, np.newaxis]
    #                 img = np.concatenate([img, img, img], axis=2)
    #             img = imresize(img, (224, 224))
    #             img = img.transpose(2, 0, 1)
    #             assert img.shape == (3, 224, 224)
    #             assert np.max(img) <= 255

    #             # Save image to HDF5 file
    #             images[i] = img

    #             for j, c in enumerate(captions):
    #                 # Encode captions
    #                 enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
    #                     word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))

    #                 # Find caption lengths
    #                 c_len = len(c) + 2

    #                 enc_captions.append(enc_c)
    #                 caplens.append(c_len)
    #             for j,c in enumerate(tags):
    #                 enc_c=[tag_map['<start>']] + [tag_map.get(word, tag_map['<unk>']) for word in c] + [
    #                     tag_map['<end>']] + [tag_map['<pad>']] * (max_len - len(c))
    #                 c_len=len(c)+2
    #                 enc_tags.append(enc_c)
    #                 taglens.append(c_len)
    #         # Sanity check
    #         assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)
    #         print("images len:", len(images))
    #         input("Please press the Enter key to proceed")

    #         # Save encoded captions and their lengths to JSON files
    #         with open(os.path.join(output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(enc_captions, j)

    #         with open(os.path.join(output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(caplens, j)

    #         with open(os.path.join(output_folder, split + '_TAGS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(enc_tags, j)

    #         with open(os.path.join(output_folder, split + '_TAGLENS_' + base_filename + '.json'), 'w') as j:
    #             json.dump(taglens, j)

    #         with open(os.path.join(output_folder, split + '_TAGSYN_' + base_filename + '.json'), 'w') as j:
    #             json.dump(img_tags_yn, j)



def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(dataname, model, epoch, encoder, decoder, encoder_optimizer, decoder_optimizer, fold_id=None):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    # print("1new_utils")
    state = {'epoch': epoch,
             # 'epochs_since_improvement': epochs_since_improvement,
             # 'bleu-4': bleu4,
             # 'f1': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    floder = './checkpoint/'
    filename = model + 'checkpoint_' + dataname + '.pth.tar'
    if fold_id is not None:
        torch.save(state, floder + 'fold' + str(fold_id) + '_' + str(epoch) + filename)
    else:
        torch.save(state, floder + str(epoch) + filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    # if is_best:
    #     torch.save(state, floder + 'BEST' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)
