from sklearn import datasets
import torch
from torch.utils.data import Dataset
import h5py
import json
import os


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, max_len, max_tag_len, split, prefix, fold_id, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split

        if fold_id is not None:
            self.fold_id = str(fold_id)+'_'
        else:
            self.fold_id = ''
        self.max_len = max_len
        self.max_tag_len = max_tag_len
        self.prefix = prefix
        dataname = self.fold_id + self.prefix
        # if self.split is 'TRAIN':
        #     dataname += '_TRAIN'

        # Open hdf5 file where images are stored
        self.h = h5py.File(os.path.join(data_folder, dataname + 'IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']
        # print("cpi:", self.cpi)

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, dataname + 'CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, dataname + 'CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)
        
        with open(os.path.join(data_folder, dataname+'TAGS_'+data_name+'.json'),'r') as j:
            self.tags=json.load(j)
        
        with open(os.path.join(data_folder,dataname+'TAGSYN_'+data_name+'.json'),'r') as j:
            self.tags_yn=json.load(j)
        
        with open(os.path.join(data_folder,dataname+'TAGLENS_'+data_name+'.json'),'r') as j:
            self.taglens=json.load(j)

        # with open(os.path.join(data_folder,self.split+'_TAGSYN_'+data_name+'.json'),'r') as j:
        #     self.tags_yn=json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

        self.long_cap_num = 0
        
        word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
        with open(word_map_file, 'r') as j:
            self.word_map = json.load(j)  #1198, dict{word:No.}

        tag_map_file=os.path.join(data_folder,'TAGMAP_'+data_name+'.json')
        with open(tag_map_file,'r') as j:
            self.tag_map=json.load(j)


    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            img = self.transform(img)

        caplen = self.caplens[i]
        # print(caplen)
        # if caplen > self.max_len:
        #     caplen = 0
        ## if self.split is 'TRAIN' and caplen > self.max_len:
        ##     caplen = self.max_len+2
            # self.long_cap_num += 1
            # # print(self.long_cap_num)
            # return None
        caplen = torch.LongTensor([caplen])

        
        caption = self.captions[i]
        # print(i, "-:", caption) #i:标号，不按顺序。caption：第i个数据的caption，tensor类型
        
        ## drop = 0
        ## if len(caption) > self.max_len+2:
        ##     caption = caption[:self.max_len+1]
        ##     caption.append(self.word_map['<end>'])
        ##     drop = 1
        ## while len(caption) < self.max_len:
        ##     caption.append(self.word_map['<pad>'])
        
        #    if self.split is 'TRAIN':
            #     drop = True
        # if self.split is 'TRAIN':
        #     enc_c = [self.word_map['<start>']] + [self.word_map.get(word, self.word_map['<unk>']) for word in caption] + [
        #             self.word_map['<end>']] + [self.word_map['<pad>']] * (self.max_len - len(caption))
        caption = torch.LongTensor(caption)

        
        tag=self.tags[i]
        # if self.split is 'TRAIN':
        #     if len(tag) > self.max_tag_len:
        #         tag = tag[:self.max_tag_len]
        #     enc_t=[self.tag_map['<start>']] + [self.tag_map.get(word, self.tag_map['<unk>']) for word in tag] + [
        #                 self.tag_map['<end>']] + [self.tag_map['<pad>']] * (self.max_tag_len - len(tag))
        #     tag = torch.LongTensor(enc_t)
        if len(tag) > self.max_tag_len:
            tag = tag[:self.max_tag_len+1]
            tag.append(self.tag_map['<end>'])
        tag = torch.LongTensor(tag)
        
        taglen=torch.LongTensor([self.taglens[i]])

        tags_yn=torch.LongTensor(self.tags_yn[i])

        if self.split is 'TRAIN':
            return img, caption, caplen,tag,taglen,tags_yn
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)]
            all_captions = torch.LongTensor(all_captions)
            # print("all_captions:", all_captions)
            return img, caption, caplen, all_captions,tag,taglen,tags_yn

    def __len__(self):
        return self.dataset_size
