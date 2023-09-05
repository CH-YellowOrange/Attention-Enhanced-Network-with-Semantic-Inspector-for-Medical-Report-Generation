from mmap import ACCESS_WRITE
from cv2 import floodFill
from sklearn import metrics
from torch import long
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils import data
import torch.utils.data
from torch.utils.data import dataloader
import torchvision.transforms as transforms
from datasets import *
from new_utils import *
# from nltk.translate.bleu_score import corpus_bleu
# import torch.nn.functional as F
from tqdm import tqdm
from nlgeval import NLGEval
# from sklearn.metrics import f1_score
import numpy as np
import random
torch.set_printoptions(profile="full")

# Parameters
data_folder = './iu_10fold'  # folder with data files saved by create_input_files.py
# data_name = 'coco_1_cap_per_img_5_min_word_freq'  # base name shared by data files
data_name = 'iu'  # base name shared by data files
# checkpoint = 'BEST_checkpoint_coco_1_cap_per_img_5_min_word_freq.pth.tar'  # model checkpoint
word_map_file = data_folder + '/WORDMAP_' + data_name + '.json'  # word map, ensure it's the same the data was encoded with and the model was trained with
tag_map_file = data_folder + '/TAGMAP_' + data_name + '.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
lbda=0.5
num_heads = 8
max_len = 50
max_tag_len = 20


def tag_origin(tag):
    tag_ori = []
    for i in tag:
        i = i.item()
        # print(i)
        # print(type(i))
        if i != 0 and i != 211 and i != 212:
            tag_ori.append(rev_tag_map[i])
    
    return ",".join(tag_ori) #string


def tags_yn_convert(mlp_out):
    threshold = 0.3
    pred_tags_yn = torch.zeros_like(mlp_out).long()

    for i, beam in enumerate(mlp_out):
        for j, val in enumerate(beam):
            if val > threshold:
                pred_tags_yn[i][j] = 1
    
    return pred_tags_yn


def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]


def evaluate(beam_size, fold_id, f, print_head=0):
    """
    Evaluation
    :param beam_size: beam size at which to generate captions for evaluation
    :return: BLEU-4 score
    """
    # DataLoader
    test_set = []
    test_set.append(CaptionDataset(data_folder, data_name, max_len=max_len, max_tag_len=max_tag_len, split='TEST', prefix='TRAIN_', fold_id=fold_id, transform=transforms.Compose([normalize])))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(test_set),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    # TODO: Batched Beam Search
    # Therefore, do not use a batch_size greater than 1 - IMPORTANT!

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
    references = list()
    hypotheses = list()

    # For each image
    tag_ori_list = []
    heads_weight_list = []
    cur_heads_weight = torch.empty(num_heads).to(device)
    num_heads_weights = []
    cnt = 0
    tmp_str = ''
    for i, (image, caps, caplens, allcaps,tags,taglens,tags_yn) in enumerate(tqdm(loader, desc="EVALUATING AT BEAM SIZE " + str(beam_size))):
        
        tag_ori_list.append(tag_origin(tags[0]))
        k = beam_size

        # Move to GPU device, if available
        image = image.to(device)  # (1, 3, 256, 256)
        tags=tags.to(device)
        # if(i==0):
        #     print(tags)
        taglens=taglens.to(device)
        tags_yn=tags_yn.to(device)

        # Encode
        # encoder_out,h_c= encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_out,hc=encoder(image)
        # enc_image_size = encoder_out.size(1)
        encoder_dim = encoder_out.size(-1)

        # # Flatten encoding

        # We'll treat the problem as having a batch size of k
        encoder_out = encoder_out.expand(k,encoder_dim)  # (k, encoder_dim)

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        h1,c1=decoder.init_hidden_state(k)
        h2,c2=decoder.init_hidden_state(k)
        decoder_dim=h2.size(1)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>

        if print_head == 1 and i < 5:
            print(i, "-", end = ' ')
            tmp_str += str(i)+'-'
        while True:

            embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
            ctx1, heads_weight_i1= decoder.attention(encoder_out,h1)
            if print_head==1 and i<5 and not heads_weight_i1.equal(cur_heads_weight):
                # print("ctx1:", ctx1, end = ' ')
                tmp_str += 'heads_weight_i1:'+str(heads_weight_i1)+'\n'
                # print("heads_weight_i1:", heads_weight_i1)
                cur_heads_weight = heads_weight_i1
            # cnt += 1

            h1,c1=decoder.sent_lstm(torch.cat([ctx1,h1],dim=1),(h1,c1))
            h2,c2=decoder.decode_step1(torch.cat([embeddings, ctx1,h1], dim=1),(h2, c2))
            scores1=decoder.fc(h2)
            if step==1:
                scores=scores1
            else:
                scores=scores1+lbda*scores2 
            ctx2, heads_weight_i2=decoder.attention(encoder_out,h2)
            if print_head==1 and i<5 and not heads_weight_i2.equal(cur_heads_weight):
                # print("ctx2:", ctx2, end = ' ')
                tmp_str += 'heads_weight_i2:'+str(heads_weight_i2)+'\n'
                # print("heads_weight_i2:", heads_weight_i2)
                cur_heads_weight = heads_weight_i2

            h3,c3=decoder.decode_step2(torch.cat([embeddings,ctx2,h2],dim=1),(h2,c2))

            scores2=decoder.fc(h3)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words // vocab_size  # (s)
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != word_map['<end>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            ## tagembedding = tagembedding[prev_word_inds[incomplete_inds]]
            h1 = h1[prev_word_inds[incomplete_inds]]
            c1 = c1[prev_word_inds[incomplete_inds]]
            h2 = h2[prev_word_inds[incomplete_inds]]
            c2 = c2[prev_word_inds[incomplete_inds]]
            # h3 = h3[prev_word_inds[incomplete_inds]]
            # c3 = c3[prev_word_inds[incomplete_inds]]
            scores2=scores2[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)


            # Break if things have been going on too long
            if step > 50:
                break
            step += 1
            
        if len(complete_seqs_scores)==0:
            continue
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]

        # References
        # img_caps = allcaps[0].tolist()
        # img_captions = list(
        #     map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
        #         img_caps))  # remove <start> and pads
        
        # references.append(img_captions)
        # #print(references)

        # # Hypotheses
        # hypotheses.append([w for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        # img_caps = allcaps[0].tolist()
        img_caps = caps.tolist()
        # if len(img_caps) > 50:
        #     print("TOO LONG!")
        img_captions = list(
            map(lambda c: [rev_word_map[w] for w in c if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}],
                img_caps))  # remove <start> and pads
        img_caps = [' '.join(c) for c in img_captions]
        # print(img_caps)
        references.append(img_caps)
        #print(references)

        # Hypotheses
        hypothesis = ([rev_word_map[w] for w in seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}])
        hypothesis = ' '.join(hypothesis)
        # print(hypothesis)
        #print(hypothesis)
        hypotheses.append(hypothesis)
        #print(hypotheses)
        assert len(references) == len(hypotheses)

    # print("train_data len:", loader.__len__())
    # print("long cap num:", test_set.long_cap_num)
    # input("Please press the enter to proceed")
    # print(hypotheses)
    # print(references[-9])

    if print_head and f is not None:
        f.write(tmp_str)
    metrics_dict = {}
    try:
        metrics_dict = nlgeval.compute_metrics(references, hypotheses)
        for key in metrics_dict.keys():
            metrics_dict[key] = float(format(metrics_dict[key], '.5f'))
    except:
        pass
    assert len(num_heads_weights) == len(heads_weight_list)

    return metrics_dict, heads_weight_i1



if __name__ == '__main__':
    with open('./10fold_result.txt', 'a') as f:
        for fold in range(10):
            if fold not in [8]:
                continue
            print("fold =", fold)
            best_b1 = 0
            best_b2 = 0
            best_b3 = 0
            best_b4 = 0
            best_m = 0
            best_r = 0
            best_c = 0
            best_epoch = 0
            best_bs = 0
            dict_str = '\nfold'+str(fold)+'\n'
            f.write(dict_str)
            dict_str = ''
            epoch = 0
            while epoch < 5:
                try:
                    print_head = 1
                    # Load model
                    print("Epoch =", epoch)
                    checkpoint='./checkpoint/' + 'fold' + str(fold) + '_' + str(epoch) + '_vit_selfatt1_checkpoint_' + data_name + '.pth.tar'
                    # checkpoint='./checkpoint/' + str(epoch) + '_BEST_vit_selfatt_dwt_3&3_186_checkpoint_' + data_name + '.pth.tar'
                    # try:
                    checkpoint = torch.load(checkpoint,map_location = device)
                except:
                    # dict_str = ''
                    epoch += 1
                    continue

                decoder = checkpoint['decoder']
                decoder = decoder.to(device)
                decoder.eval()
                encoder = checkpoint['encoder']
                encoder = encoder.to(device)
                encoder.eval()
                dict_str += 'Epoch='+str(epoch)+':\n'

                nlgeval=NLGEval()

                # Load word map (word2ix)
                with open(word_map_file, 'r') as j:
                    word_map = json.load(j)
                with open(tag_map_file, 'r') as j:
                    tag_map = json.load(j)
                rev_word_map = {v: k for k, v in word_map.items()}
                rev_tag_map = {v: k for k, v in tag_map.items()}
                vocab_size = len(word_map)

                # Normalization transform
                normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225])
                
                for beam_size in range(1, 6):
                    print('beamsize={bs}:'.format(bs=beam_size))
                    metrics_dict, heads_weight = evaluate(beam_size, fold, f, print_head=print_head)
                    if print_head:
                        print_head = 0
                    print(heads_weight)
                    dict_str += 'beamsize='+str(beam_size)+':\n'+str(metrics_dict)+'\n'
                    if metrics_dict:
                        if metrics_dict['Bleu_1'] > best_b1:
                            best_b1 = metrics_dict['Bleu_1']
                        if metrics_dict['Bleu_2'] > best_b2:
                            best_b2 = metrics_dict['Bleu_2']
                        if metrics_dict['Bleu_3'] > best_b3:
                            best_b3 = metrics_dict['Bleu_3']
                        if metrics_dict['Bleu_4'] > best_b4:
                            best_b4 = metrics_dict['Bleu_4']
                            best_epoch = epoch
                            best_bs = beam_size
                        if metrics_dict['METEOR'] > best_m:
                            best_m = metrics_dict['METEOR']
                        if metrics_dict['ROUGE_L'] > best_r:
                            best_r = metrics_dict['ROUGE_L']
                        if metrics_dict['CIDEr'] > best_c:
                            best_c = metrics_dict['CIDEr']
                        print(metrics_dict)

                print()
                dict_str += '\n'
                f.write(dict_str)
                dict_str = ''
                epoch += 1
            
            print('b1:', best_b1, 'b2:', best_b2, 'b3:', best_b3, 'b4:', best_b4, 'm:', best_m, 'r:', best_r, 'c:', best_c)
            dict_str = str(best_b1)+' '+str(best_b2)+' '+str(best_b3)+' '+str(best_b4)+' '+str(best_m)+' '+str(best_r)+' '+str(best_c)+'\n\n'
            f.write(dict_str)
            dict_str = ''
        
