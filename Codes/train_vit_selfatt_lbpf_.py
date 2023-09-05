import time
from numpy import False_
from timm import models
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import dataloader
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from model_vit_selfatt_lbpf import Encoder, DecoderWithAttention
from datasets import *
from new_utils import *
from nltk.translate.bleu_score import corpus_bleu
import torchmetrics
# Data parameters
data_folder = './iu_10fold'  # folder with data files saved by create_input_files.py
data_name = 'iu'
model = '_vit_selfatt1_'

# Model parameters 
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
encoder_dim = 512
decoder_dim = 512  # dimension of decoder RNN
need_heads_weight=2 # 0:no head weight 1:single weight 2:double weight
need_tags_supervise=True
# dropout_name = '3&3'
max_len = 50
max_tag_len=20

dropout = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
start_epoch = 0
epochs = 5  # number of epochs to train for (if early stopping is not triggered)
# epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 16
workers = 1  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
alpha_c = 1.  # regularization parameter for 'doubly stochastic attention', as in the paper
lbda=0.5
best_bleu4 = 0.  # BLEU-4 score right now
print_freq = 100  # print training/validation stats every __ batches
fine_tune_encoder = False  # fine-tune encoder?
# checkpoint='./checkpoint/2_vit_selfatt_checkpoint.pth.tar'  # path to checkpoint, None if none
checkpoint=None
# taglen=10
def main():
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map,rev_tag_map,tag_map

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)  #1198, dict{word:No.}
    rev_word_map = {v: k for k, v in word_map.items()}

    tag_map_file=os.path.join(data_folder,'TAGMAP_'+data_name+'.json')
    with open(tag_map_file,'r') as j:
        tag_map=json.load(j)
    rev_tag_map = {v: k for k, v in tag_map.items()} #213, dict{No.:word}
    # print(rev_tag_map[1])
    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)
    criterion1= nn.MultiLabelSoftMarginLoss().to(device)
    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_set = []
    for i in range(10):
        if i not in [0,6,8]:
            continue
        
        epochs_since_improvement = 0
        train_set.clear()
        # Initialize / load checkpoint
        # checkpoint = None
        if checkpoint is None:
            decoder = DecoderWithAttention(attention_dim=attention_dim,
                                        embed_dim=emb_dim,
                                        decoder_dim=decoder_dim,
                                        vocab_size=len(word_map),
                                        tag_size=len(tag_map),
                                        encoder_dim=encoder_dim,
                                        dropout=dropout,
                                        lbda=lbda,
                                        need_heads_weight=need_heads_weight,
                                        need_tags_supervise=need_tags_supervise)
            decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                                lr=decoder_lr)
            encoder = Encoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim)
            # encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                lr=encoder_lr) if fine_tune_encoder else None

        else:
            checkpoint = torch.load(checkpoint)
            start_epoch = checkpoint['epoch'] + 1
            # epochs_since_improvement = checkpoint['epochs_since_improvement']
            # best_bleu4 = checkpoint['bleu-4']
            decoder = checkpoint['decoder']
            decoder_optimizer = checkpoint['decoder_optimizer']
            encoder = checkpoint['encoder']
            encoder_optimizer = checkpoint['encoder_optimizer']
            if fine_tune_encoder is True and encoder_optimizer is None:
                # encoder.fine_tune(fine_tune_encoder)
                encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                    lr=encoder_lr)

        # Move to GPU, if available
        decoder = decoder.to(device)
        encoder = encoder.to(device)
        
        if i < 9:
            for j in range(i):
                train_set.append(CaptionDataset(data_folder, data_name, max_len, max_tag_len, split='TRAIN', prefix='TRAIN_', fold_id=j, transform=transforms.Compose([normalize])))

            val_loader = torch.utils.data.DataLoader(
                CaptionDataset(data_folder, data_name,  max_len, max_tag_len, split='VAL', prefix='TRAIN_', fold_id=i+1, transform=transforms.Compose([normalize])), 
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            for j in range(i+2, 11):
                train_set.append(CaptionDataset(data_folder, data_name, max_len, max_tag_len, split='TRAIN', prefix='TRAIN_', fold_id=j, transform=transforms.Compose([normalize])))

        else:
            for j in range(1, i):
                train_set.append(CaptionDataset(data_folder, data_name, max_len, max_tag_len, split='TRAIN', prefix='TRAIN_', fold_id=j, transform=transforms.Compose([normalize])))

            val_loader = torch.utils.data.DataLoader(
                CaptionDataset(data_folder, data_name,  max_len, max_tag_len, split='VAL', prefix='TRAIN_', fold_id=0, transform=transforms.Compose([normalize])), 
                batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            for j in range(i+1, 11):
                train_set.append(CaptionDataset(data_folder, data_name, max_len, max_tag_len, split='TRAIN', prefix='TRAIN_', fold_id=j, transform=transforms.Compose([normalize])))

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.dataset.ConcatDataset(train_set),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        
        # Epochs
        for epoch in range(start_epoch, epochs):
            # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
            if epochs_since_improvement == 50:
                break
            if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
                adjust_learning_rate(decoder_optimizer, 0.8)
                if fine_tune_encoder:
                    adjust_learning_rate(encoder_optimizer, 0.8)

            # One epoch's training
            train(train_loader=train_loader,
                encoder=encoder,
                decoder=decoder,
                criterion=criterion,
                criterion1=criterion1,
                encoder_optimizer=encoder_optimizer,
                decoder_optimizer=decoder_optimizer,
                epoch=epoch)
            
            print()

            # One epoch's validation
            recent_bleu4 = validate(val_loader=val_loader,
                                    encoder=encoder,
                                    decoder=decoder,
                                    criterion=criterion,
                                    criterion1=criterion1)
            # recent_bleu4=eval_lbpf.evaluate(beam_size=batch_size,
            #                              data_loader=val_loader,
            #                              encoder=encoder,decoder=decoder)

            # Check if there was an improvement
            is_best = recent_bleu4 > best_bleu4
            best_bleu4 = max(recent_bleu4, best_bleu4)
            if not is_best:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            else:
                epochs_since_improvement = 0

            # Save checkpoint
            save_checkpoint(data_name, model, epoch, encoder, decoder, encoder_optimizer,
                            decoder_optimizer, fold_id=i)


def train(train_loader, encoder, decoder, criterion,criterion1, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """

    decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses1 = AverageMeter()  # loss1 (per word decoded)
    losses2 = AverageMeter()  # loss2 (per word decoded)
    lossest = AverageMeter()  # losst (per tag decoded)
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    # tags: 2d list, tags[i]: tags of image i represented by tag_map
    # tags_yn: 2d list, tags[i]: tags of image i represented by tag_map(one-hot)
    # tag_len_max = 0 #36
    for i, (imgs, caps, caplens,tags,taglens,tags_yn) in enumerate(train_loader):
        '''
        for batch in tags:
            for j, val in enumerate(batch):
                if val == 0:
                    if j > tag_len_max:
                        tag_len_max = j
                    break
        '''

        data_time.update(time.time() - start)

        # Move to GPU, if available
        # print(taglens.shape)
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)

        tags=tags.to(device)
        taglens=taglens.to(device)

        tags_yn=tags_yn.to(device)

        # Forward prop.
        imgs, hc = encoder(imgs)
        scores,scores1,caps_sorted, decode_lengths, alphas, sort_ind, tag_pred_out, heads_weight_i = decoder(imgs, caps, caplens,tags_yn,rev_tag_map,tag_map,hc)

        scores_copy=scores
        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        word_len=scores.size(1)
        # print(word_len)
        for k in range(1,word_len):
            scores[:,k,:]=scores[:,k,:]+ lbda*scores1[:,k-1,:]

        # print(scores.shape)
        # print(scores_copy.shape)
        targets = caps_sorted[:, 1:]
        # print(targets.shape)
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores= pack_padded_sequence(scores, decode_lengths, batch_first=True).data
        targets= pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        scores_copy= pack_padded_sequence(scores_copy, decode_lengths, batch_first=True).data
        scores1= pack_padded_sequence(scores1, decode_lengths, batch_first=True).data
        # Calculate loss
        # print(mlp_out.shape)
        # print(tags_yn.shape)
        loss1 = criterion(scores_copy, targets)
        loss2 = criterion(scores1,targets)
        ## loss3=criterion1(mlp_out,tags_yn)
        loss = loss1 + lbda*loss2

        if need_tags_supervise:
            loss4 = criterion1(tag_pred_out, tags_yn)
            loss += 5*loss4
        # Add doubly stochastic attention regularization
        # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        # print(sum(decode_lengths), sum(taglens).item())
        top5 = accuracy(scores, targets, 5)
        losses1.update(loss1.item(), sum(decode_lengths))
        losses2.update(loss2.item(), sum(decode_lengths))
        losses.update(loss.item(), sum(decode_lengths))
        if need_tags_supervise:
            lossest.update(loss4.item(), sum(taglens).item())
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)
        ## mlp_acc=torchmetrics.functional.accuracy(mlp_out,tags_yn)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Top-5 Accuracy {top5.val:.3f} (avg:{top5.avg:.3f})'
                  ## 'mlp Accuracy {mlp_acc:.4f} ({mlp_acc:.4f})\t'
                                                                        .format(epoch, i, len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          ## mlp_acc=mlp_acc,
                                                                          top5=top5accs), end = '\t')
            if need_tags_supervise:
                print('Loss1 {loss1.val:.4f} (avg:{loss1.avg:.4f})\t'
                      'Loss2 {loss2.val:.4f} (avg:{loss2.avg:.4f})\t'
                      'Loss_t {loss_t.val:.4f} (avg:{loss_t.avg:.4f})\t'
                      'Loss {loss.val:.4f} (avg:{loss.avg:.4f})'.format(loss1=losses1, loss2=losses2, loss_t=lossest, loss=losses))
            else:
                print('Loss1 {loss1.val:.4f} (avg:{loss1.avg:.4f})\t'
                      'Loss2 {loss2.val:.4f} (avg:{loss2.avg:.4f})\t'
                      'Loss {loss.val:.4f} (avg:{loss.avg:.4f})'.format(loss1=losses1, loss2=losses2, loss=losses))

    # print("tag_len_max:", tag_len_max)

def validate(val_loader, encoder, decoder, criterion,criterion1):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score

    """
    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = list()  # references (true captions) for calculating BLEU-4 score
    hypotheses = list()  # hypotheses (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, allcaps,tags,taglens,tags_yn) in enumerate(val_loader):
            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            tags=tags.to(device)
            taglens=taglens.to(device)
            tags_yn=tags_yn.to(device)
            # Forward prop.
            if encoder is not None:
                imgs, hc = encoder(imgs)

            scores, scores1, caps_sorted, decode_lengths, alphas, sort_ind, tag_pred_out, _ = decoder(imgs, caps, caplens, tags_yn, rev_tag_map,tag_map,hc)

            word_len=scores.size(1)
            scores_copy = scores.clone()
            scores_copy1 = scores.clone()
            for k in range(1,word_len):
                scores[:,k,:]=scores[:,k,:]+scores1[:,k-1,:]

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]


            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores= pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets= pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            scores_copy= pack_padded_sequence(scores_copy, decode_lengths, batch_first=True).data
            scores1= pack_padded_sequence(scores1, decode_lengths, batch_first=True).data
            # Calculate loss
            loss1 = criterion(scores_copy, targets)
            loss2= criterion(scores1,targets)
            ## loss3=criterion1(mlp_out,tags_yn)
            loss = loss1 + lbda*loss2
            if need_tags_supervise:
                loss4 = criterion1(tag_pred_out, tags_yn)
                loss += 5*loss4
    
            # Add doubly stochastic attention regularization
            # loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores, targets, 5)
           
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            ## mlp_acc=torchmetrics.functional.accuracy(mlp_out,tags_yn)

            start = time.time()


            if i % print_freq == 0:

                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      ## 'mlp Accuracy {mlp_acc:.4f} ({mlp_acc:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                                loss=losses,
                                                                                ## mlp_acc=mlp_acc,
                                                                                top5=top5accs))

            # Store references (true captions), and hypothesis (prediction) for each image
            # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
            # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

            # References
            allcaps = allcaps[sort_ind]  # because images were sorted in the decoder
            for j in range(allcaps.shape[0]):
                img_caps = allcaps[j].tolist()
                img_captions = list(
                    map(lambda c: [w for w in c if w not in {word_map['<start>'], word_map['<pad>']}],
                        img_caps))  # remove <start> and pads
                references.append(img_captions)

            # Hypotheses
            _, preds = torch.max(scores_copy1, dim=2)
            preds = preds.tolist()
            temp_preds = list()
            for j, p in enumerate(preds):
                temp_preds.append(preds[j][:decode_lengths[j]])  # remove pads
            preds = temp_preds
            hypotheses.extend(preds)

            assert len(references) == len(hypotheses)

        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypotheses)


        print(
            '\n * LOSS - {loss.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss=losses,
                top5=top5accs,
                ## mlp_acc=mlp_acc,
                bleu=bleu4))

    return bleu4


if __name__ == '__main__':
    main()
