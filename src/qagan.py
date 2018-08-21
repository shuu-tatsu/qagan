import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import generator
import discriminator
import config
import random


#os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
#parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)

#img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
    def __init__(self, input_lang, output_lang):
        super(Generator, self).__init__()
        hidden_size = 100
        self.encoder = generator.EncoderRNN(input_lang.n_words, hidden_size).to(device)
        self.decoder = generator.AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    def forward(self, a_tensor):

        decoded_index_list = []

        input_tensor = a_tensor
        encoder_hidden = self.encoder.initHidden()
        input_length = input_tensor.size(0)
        #target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(config.args.max_length, self.encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[config.args.sos_token]], device=device)

        decoder_hidden = encoder_hidden

        # Without teacher forcing: use its own predictions as the next input
        for di in range(config.args.max_length):
            decoder_output, decoder_hidden, decoder_attention = self.decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
 

            if topi.item() == config.args.eos_token:
                decoded_index_list.append(config.args.eos_token)
                break
            else:
                decoded_index_list.append(topi.item())

            decoder_input = topi.squeeze().detach()

        #return img
        decoded_index_list_duble = [[index] for index in decoded_index_list] 
        gen_q_tensor = torch.LongTensor(decoded_index_list_duble).to(config.device)
        return gen_q_tensor

    def inference(self, pairs):
        bleu_score = generator.evaluateRandomly(self.encoder, self.decoder, n=10, data_pairs=pairs)
        return bleu_score


class Discriminator(nn.Module):
    def __init__(self, input_lang):
        super(Discriminator, self).__init__()
        hidden_size = config.args.embedding_dim
        vocab = input_lang
        self.encoder = discriminator.EncoderRNN(vocab,
                                  hidden_size,
                                  glove_file=config.glove_file,
                                  embedding_dim=config.args.embedding_dim).to(config.device)
        self.model = discriminator.Classifier(self.encoder, self.encoder.hidden_size).to(config.device)

    #def forward(self, img):
    def forward(self, real_q_tensor, gen_q_tensor):
        #y = model.forward(answer_tensor, question_tensor)
        y_tensor = self.model.forward(real_q_tensor, gen_q_tensor)
        validity_tensor = y_tensor.view(-1, 1)
        return validity_tensor

class Dataloader():
    def __init__(self, data_file):
        self.data_file = data_file

    def load_quora(self, input_lang):
        _, self.tensor_pairs = discriminator.get_positive_pairs(self.data_file, 'dev', input_lang)
        # self.tensor_pairs: positive_index_tuple_labeled
        # questions_index_list, answers_index_list

    def load_marco(self):
        input_lang, output_lang, train_pairs = generator.prepareData('eng_train', 'fra_train', True) 
        train_pairs_array = np.array(train_pairs).T
        pairs_list = []
        for i in range(len(train_pairs_array[0])):
            pair = [train_pairs_array[0][i], train_pairs_array[1][i]]
            pairs_list.append(generator.tensorsFromPair(pair))
        '''
        pairs_list:
        [[input_tensor, target_tensor],
         [input_tensor, target_tensor],
         ...
         [input_tenosr, target_tensor]]
        '''

        self.tensor_pairs = pairs_list
        self.train_pairs = train_pairs

        return input_lang, output_lang, train_pairs

    def iterate(self):
        return self.tensor_pairs


def get_iter_quora(tensor_pairs):
    i = random.choice([i for i in range(len(tensor_pairs[0]))])
    real_q_index_list_1 = tensor_pairs[0][i]
    real_q_index_list_2 = tensor_pairs[1][i]
    real_q_tensor_1 = torch.LongTensor(real_q_index_list_1).view(-1, 1).to(config.device)
    real_q_tensor_2 = torch.LongTensor(real_q_index_list_2).view(-1, 1).to(config.device)
    return real_q_tensor_1, real_q_tensor_2


# Dataloader
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
marco_loader = Dataloader('****')
quora_loader = Dataloader(config.quora_dev_file)


input_lang, output_lang, train_pairs = marco_loader.load_marco() # marco for generator and discriminator
quora_loader.load_quora(input_lang) # quora for discriminator

# Loss function
adversarial_loss = torch.nn.BCEWithLogitsLoss()

# Initialize generator and discriminator
generator_model = Generator(input_lang, output_lang)
discriminator_model = Discriminator(input_lang)

# Optimizers
optimizer_G = torch.optim.Adam(generator_model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator_model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

if cuda:
    generator_model.cuda()
    discriminator_model.cuda()
    adversarial_loss.cuda()

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    batch_size = 32
    batch_g_loss = 0.0
    batch_d_loss = 0.0

    #for i, (imgs, _) in enumerate(dataloader):
    for i, (q_tensor, a_tensor) in enumerate(marco_loader.iterate()):
        valid_tensor = Variable(Tensor(2, 1).fill_(1.0), requires_grad=False)
        fake_tensor = Variable(Tensor(2, 1).fill_(0.0), requires_grad=False)

        real_q_tensor_1, real_q_tensor_2 = get_iter_quora(quora_loader.iterate())

        # -----------------
        #  Train Generator
        # -----------------
        gen_q_tensor = generator_model(a_tensor)

        g_loss = adversarial_loss(discriminator_model(q_tensor, gen_q_tensor), valid_tensor)
        batch_g_loss += g_loss

        if i % batch_size == 0:
            optimizer_G.zero_grad()
            batch_g_loss.backward()
            optimizer_G.step()
            batch_g_loss = 0.0

        # ---------------------
        #  Train Discriminator
        # ---------------------

        real_loss = adversarial_loss(discriminator_model(real_q_tensor_1, real_q_tensor_2), valid_tensor)
        fake_loss = adversarial_loss(discriminator_model(q_tensor, gen_q_tensor.detach()), fake_tensor)
        d_loss = (real_loss + fake_loss) / 2
        batch_d_loss += d_loss

        if i % batch_size == 0:
            optimizer_D.zero_grad()
            batch_d_loss.backward()
            optimizer_D.step()
            batch_d_loss = 0.0

            # bleu_score = generator_model.inference(marco_loader.train_pairs)
            bleu_score = 0.0
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [Bleu: %f]" % (
                   epoch, opt.n_epochs, i, len(marco_loader.iterate()), d_loss.item(), g_loss.item(), bleu_score))
