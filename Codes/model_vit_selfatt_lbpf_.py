import imp
from json import decoder
import math
# from cv2 import hconcat
from timm.models import visformer
import torch
from torch import nn
# from torch.nn import MultiheadAttention
from torch.nn.functional import threshold
import torchvision
import json
import os
from MultiHeadAttention_ import MultiheadAttention_

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoder_dim, decoder_dim, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # vgg19 = torchvision.models.vgg19(pretrained=True)  # pretrained ImageNet ResNet-101
        # modules=list(vgg19.children())[:-2]
        #resnet=torchvision.models.resnet101(pretrained=True)
        self.vit=timm.create_model('convit_base',pretrained=True,num_classes=0) # 获取池化后未分类特征
        self.linear=nn.Linear(768,encoder_dim)
        
        self.global_cap=nn.LSTM(encoder_dim,decoder_dim,batch_first=True,num_layers=1)

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out=self.vit(images)
        # for x in out:
        #     print(x.shape)

        out=self.linear(out) #(batch_size, encoder_dim)
        return out, out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[8:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Mlp(nn.Module):
    def __init__(self,encoder_dim,tag_size):
        super(Mlp,self).__init__()
        self.linear=nn.Linear(encoder_dim,encoder_dim)
        self.linear2=nn.Linear(encoder_dim,encoder_dim)
        self.linear3=nn.Linear(encoder_dim,tag_size)
        self.softmax=nn.Softmax(dim=1)
    
    def forward(self,encoder_out):
        out=self.linear(encoder_out)
        out=self.linear2(out)
        out=self.linear3(out)
        out=self.softmax(out)
        # print("mlp out shape:", out.shape) #(batch_size, 209), out: values range (0, 1)
        # print("mlp out:", out)
        return out

class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim, embed_dim, need_heads_weight=0):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.w_q = nn.Linear(decoder_dim, attention_dim) # q is caculated by hidden dim h
        self.w_ki = nn.Linear(encoder_dim, attention_dim)  # k, v are caculated by encoder output(image features)
        self.w_vi = nn.Linear(encoder_dim, attention_dim)
        self.w_kt = nn.Linear(embed_dim, attention_dim)  # k, v are caculated by tag embedding
        self.w_vt = nn.Linear(embed_dim, attention_dim)
        self.multihead_att = MultiheadAttention_(attention_dim, num_heads=8, need_heads_weight=need_heads_weight)
        self.fc = nn.Linear(attention_dim, attention_dim)
        self.softmax=nn.Softmax(dim=1)
        self.need_heads_weight=need_heads_weight
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        # decoder_hidden: (batch_size_t,hidden_size = decoder_dim)
        # encoder_out: (batch_size_t, encoder_dim(512))
        # tag_embedding: (batch_size_t,max_tag_length(10),embed_dim)

        # dimension transfer: 1. add an dimension(1) as the first dimension of decoder_hidden;
        # 2. make batch_size_t in dimension 1 for encoder_out&tag_embedding
        # print("decoder_hidden shape:", decoder_hidden.shape)
        # print("encoder_out shape:", encoder_out.shape)
        ##print("tag_embedding shape:", tag_embedding.shape)

        decoder_hidden = torch.unsqueeze(decoder_hidden, dim = 0)
        # encoder_out = encoder_out.permute(1, 0, 2)
        # tag_embedding = tag_embedding.permute(1, 0, 2)
        w_q = self.w_q(decoder_hidden)
        w_ki = self.w_ki(encoder_out)
        w_vi = self.w_vi(encoder_out)

        att_output_i, heads_weight_i = self.multihead_att(w_q, w_ki, w_vi, need_weights=False) # need_weights指示是否需要attn_output_weight，不是multi head的weight
        ## att_output_t, _, heads_weight_t = self.multihead_att(w_q, w_kt, w_vt)
        att_output_i = torch.squeeze(att_output_i, dim = 0) # (batch_size_t, attention_dim)
        ## att_output_t = torch.squeeze(att_output_t, dim = 0) # (batch_size_t, attention_dim)
        # print("shape of att_output_i:", att_output_i.shape)
        # print("shape of att_output_t:", att_output_t.shape)
        att_output_i = self.softmax(att_output_i)
        ## att_output_t = self.softmax(att_output_t)
        ## ctx = self.fc(self.dropout(torch.cat([att_output_i, att_output_t], dim = 1)))
       
        # att_output_i = self.dropout(att_output_i)
        ctx = self.fc(att_output_i)
        # print("shape of ctx:", ctx.shape)
        # ctx1 = self.fc2(ctx.permute(1, 2, 0)).squeeze(2) # (batch_size_t, attention_dim)
        return ctx, heads_weight_i


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim,embed_dim, decoder_dim, vocab_size,tag_size,
     encoder_dim=512, dropout=0.5, lbda=0.5, need_heads_weight=0, need_tags_supervise=False):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.tag_size=tag_size
        self.need_tags_supervise=need_tags_supervise
        self.lbda = lbda

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim, embed_dim, need_heads_weight=need_heads_weight)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer

        self.tagembedding=nn.Embedding(tag_size,embed_dim)
        self.dropout = nn.Dropout(p=self.dropout)

        self.decode_lstm1=nn.LSTMCell(embed_dim,decoder_dim,bias=True)

        self.sent_lstm=nn.LSTMCell(attention_dim+decoder_dim,decoder_dim,bias=True)

        self.pf_lstm = nn.LSTMCell(decoder_dim +encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.decode_step1 = nn.LSTMCell(embed_dim + attention_dim +decoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.decode_step2 = nn.LSTMCell(embed_dim + attention_dim +decoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary

        self.fc2=nn.Linear(decoder_dim,vocab_size)

        self.global_tag=nn.LSTM(embed_dim,decoder_dim,batch_first=True,num_layers=1)
        self.mlp=Mlp(encoder_dim,tag_size-4)
        self.tag_pred_lstm = nn.LSTM(embed_dim, embed_dim, batch_first=True, num_layers = 1)
        self.tag_pred_linear = nn.Linear(embed_dim, tag_size-4)
        self.softmax = nn.Softmax(dim=1)
        self.dropout2 = nn.Dropout(p=0.3)
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self,batch_size):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        h = torch.zeros(batch_size,self.decoder_dim).to(device)  # (batch_size, decoder_dim)
        c = torch.zeros(batch_size,self.decoder_dim).to(device)
        return h, c
    
    def tag_convert(self,mlp_out,rev_tag_map,word_map,tag_len=8):
        threshold=0.3
        encoded_tag=[]
        for i,batch in enumerate(mlp_out):
            tag_list=[]
            for j,val in enumerate(batch):
                if val>threshold:
                    # print(j)
                    # print(val)
                    tag_list.append(rev_tag_map[j])
            # print(i)
            # print(len(tag_list))
            # print(tag_list)
            if len(tag_list)==0:
                enc_c=[word_map['<start>']] +[word_map['<end>']] + [word_map['<pad>']] * (tag_len - len(tag_list))
            else:
                enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in tag_list] +[word_map['<end>']] + [word_map['<pad>']] * (tag_len - len(tag_list))
            encoded_tag.append(enc_c)
        encoded_tag = torch.LongTensor(encoded_tag).to(device) #(batch_size,12)
        # print(encoded_tag.shape)
        return encoded_tag

    def forward(self, encoder_out, encoded_captions, caption_lengths,tags_yn,rev_tag_map,word_map,hc):
        """
        Forward propagation.
        :vit:encoder_out (batch_size,hiddensize)
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        tag_size=self.tag_size

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        # print("sort_ind:", sort_ind)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        tags_yn = tags_yn[sort_ind]
        #addtagimformation
        # taglens.sort_ind=taglens.squeeze(1).sort(dim=0,descending=True)
        # encoded_tags=encoded_tags[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        
        # Initialize LSTM state
        h1,c1=self.init_hidden_state(batch_size)
        h2,c2=self.init_hidden_state(batch_size)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()
        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        predictions2 = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        num_pixels=196
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            ctx, _=self.attention(encoder_out[:batch_size_t],h1[:batch_size_t])
            h1,c1=self.sent_lstm(torch.cat([ctx,h1[:batch_size_t]],dim=1),(h1[:batch_size_t],c1[:batch_size_t]))
            # print(ctx.shape)
            # print(embeddings.shape)
            # print(batch_size_t, t, embeddings.shape, ctx.shape, h1.shape)
            h2, c2 = self.decode_step1(
                torch.cat([embeddings[:batch_size_t, t, :], ctx[:batch_size_t],h1[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))  # (batch_size_t, decoder_dim)
            h2 = self.dropout(h2)
            preds = self.fc(h2)  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

            ctx2, heads_weight_i=self.attention(encoder_out[:batch_size_t],h2[:batch_size_t])
            
            h3,c3=self.decode_step2(torch.cat([embeddings[:batch_size_t,t,:],ctx2[:batch_size_t],h2[:batch_size_t]], dim=1),
                (h2[:batch_size_t], c2[:batch_size_t]))
            h3 = self.dropout(h3)
            preds2 = self.fc2(h3)  # (batch_size_t, vocab_size)
            predictions2[:batch_size_t, t, :] = preds2
            # alphas[:batch_size_t, t, :] = alpha
        
        tag_pred_out=None
        if self.need_tags_supervise:
            pred_word_ind = (predictions+self.lbda*predictions2).argmax(dim=2)
            pred_word_emb = self.embedding(pred_word_ind)
            tag_pred_out, _ = self.tag_pred_lstm(pred_word_emb)
            tag_pred_out = tag_pred_out[:, -1, :].squeeze(1)
            tag_pred_out = self.dropout2(tag_pred_out)
            tag_pred_out = self.softmax(self.tag_pred_linear(tag_pred_out))
            # print("tag_pred_out shape:", tag_pred_out.shape)

        return predictions,predictions2, encoded_captions, decode_lengths, alphas, sort_ind, tag_pred_out, heads_weight_i