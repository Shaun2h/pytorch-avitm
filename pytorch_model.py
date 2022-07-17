import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import math


class ProdLDA(nn.Module):

    def __init__(self, inp_dict):
        super(ProdLDA, self).__init__()
        inp_dim,en1,en2,num_topic,init_mult,variance,num_input = self.starterdict_parser(inp_dict)
        self.num_topic = num_topic
        # encoder
        
        self.en1_fc     = nn.Linear(num_input, en1)             # 1995 -> 100
        self.en2_fc     = nn.Linear(en1, en2)             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_fc    = nn.Linear(en2, num_topic)             # 100  -> 50
        self.mean_bn    = nn.BatchNorm1d(num_topic)                      # bn for mean
        self.logvar_fc  = nn.Linear(en2, num_topic)             # 100  -> 50
        self.logvar_bn  = nn.BatchNorm1d(num_topic)                      # bn for logvar
        # z
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        self.decoder    = nn.Linear(num_topic, num_input)             # 50   -> 1995
        self.decoder_bn = nn.BatchNorm1d(num_input)                      # bn for decoder
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, num_topic).fill_(0)
        prior_var    = torch.Tensor(1, num_topic).fill_(variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar) # buffers are just old storage, and are not updated with backpropagation.
        
        # initialize decoder weight
        if init_mult != 0:
            #std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            self.decoder.weight.data.uniform_(0, init_mult)
        # remove BN's scale parameters
        # self.logvar_bn.register_parameter('weight', None)
        # self.mean_bn.register_parameter('weight', None) # This causes errors because you can't have a none weight.
        # self.decoder_bn.register_parameter('weight', None)
        # self.decoder_bn.register_parameter('weight', None)

    def forward(self, input, compute_loss=False, avg_loss=True):
        # compute posterior
        en1 = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2 = F.softplus(self.en2_fc(en1))                              # encoder2 output
        en2 = self.en2_drop(en2)
        posterior_mean   = self.mean_bn  (self.mean_fc  (en2))          # posterior mean
        posterior_logvar = self.logvar_bn(self.logvar_fc(en2))          # posterior log variance
        # print(posterior_logvar)
        posterior_var    = posterior_logvar.exp()
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z)                                                # mixture probability
        p = self.p_drop(p)
        # do reconstruction
        recon = F.softmax(self.decoder_bn(self.decoder(p)))             # reconstructed distribution over vocabulary

        if compute_loss:
            return recon, self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss), p
        else:
            return recon,p

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        # print("input:",input.shape)
        # print("recon:",recon.shape)
        # print("NL:",input * (recon+1e-10).log().shape)
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.num_topic )
        # loss

        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean().reshape((-1))
        else:
            return loss

    def starterdict_parser(self,inp_dict):
        if "inp_dim" in inp_dict:
            inp_dim = inp_dict["inp_dim"]
        else:
            inp_dim = 1995
        if "en1" in inp_dict:
            en1 = inp_dict["en1"]
        else:
            en1 = 100
        if "en2" in inp_dict:
            en2 = inp_dict["en2"]
        else:
            en2 = 100
        
        if "num_topic" in inp_dict:
            num_topic = inp_dict["num_topic"]
        else:
            num_topic = 50
        
        if "init_mult" in inp_dict:
            init_mult = inp_dict['init_mult']
        else:
            init_mult = 1.0
            
        if "variance" in inp_dict:
            variance = inp_dict["variance"]
        else:
            variance = 0.995
        if "num_input" in inp_dict:
            num_input = inp_dict["num_input"]
        else:
            raise ValueError("Input dict did not specify starting layer's number of inputs. Can't determine shape.")
            
        return inp_dim,en1,en2,num_topic,init_mult,variance, num_input
        
        

class Hypernet_LDA(nn.Module):

    def __init__(self, inp_dict):
        super(Hypernet_LDA, self).__init__()
        inp_dim,en1,en2,num_topic,init_mult,variance,num_input = self.starterdict_parser(inp_dict)
        self.num_topic = num_topic
        # encoder
        self.referraldict = inp_dict
        
        
        # list of buffers to generate -> 
        # en1_fc     = torch.Tensor(num_input, en1)
        # en1_bias = torch.Tensor(en1)
        # en2_fc     = torch.Tensor(en1, en2)
        # en2_bias = torch.Tensor(en2)
        # mean_fc    = torch.Tensor(en2, num_topic)
        # mean_bias = torch.Tensor(num_topic)
        # logvar_fc  = torch.Tensor(en2, num_topic)
        # logvar_bias = torch.Tensor(num_topic)
        # decoder = torch.Tensor(num_topic, num_input)
        # decoder_bias = torch.Tensor(num_input)
        
        self.hyp_en1_fc_1 = torch.nn.Linear(num_input, 128)
        self.hyp_en1_fc_2 = torch.nn.Linear(128, num_input*en1)
        self.hyp_en1_fc_b_1 = torch.nn.Linear(128, 24)
        self.hyp_en1_fc_b_1_dropout = torch.nn.Dropout(0.2)
        self.hyp_en1_fc_b_2 = torch.nn.Linear(24, 1)
        
        self.hyp_en2_fc_1 = torch.nn.Linear(128, 128)
        self.hyp_en2_fc_2 = torch.nn.Linear(128, en1*en2)
        self.hyp_en2_fc_b_1 = torch.nn.Linear(128, 24)
        self.hyp_en2_fc_b_1_dropout = torch.nn.Dropout(0.2)
        self.hyp_en2_fc_b_2 = torch.nn.Linear(24, 1)
        
        self.hyp_mean_fc_1 = torch.nn.Linear(128, 128)
        self.hyp_mean_fc_2 = torch.nn.Linear(128, en2*num_topic)
        self.hyp_mean_fc_b_1 = torch.nn.Linear(128, 24)
        self.hyp_mean_fc_b_1_dropout = torch.nn.Dropout(0.2)
        self.hyp_mean_fc_b_2 = torch.nn.Linear(24, 1)
        
        self.hyp_logvar_fc_1 = torch.nn.Linear(128, 128)
        self.hyp_logvar_fc_2 = torch.nn.Linear(128, en2*num_topic)
        self.hyp_logvar_fc_b_1 = torch.nn.Linear(128, 24)
        self.hyp_logvar_fc_b_1_dropout = torch.nn.Dropout(0.2)
        self.hyp_logvar_fc_b_2 = torch.nn.Linear(24, 1)
        
        self.hyp_decoder_fc_1 = torch.nn.Linear(128*2, 128) # take in logvar and mean's intermediates.
        self.hyp_decoder_fc_2 = torch.nn.Linear(128, num_input*num_topic) # might have to recurrent neural net this or something...
        self.hyp_decoder_fc_b_1 = torch.nn.Linear(128, 24)
        self.hyp_decoder_fc_b_1_dropout = torch.nn.Dropout(0.2)
        self.hyp_decoder_fc_b_2 = torch.nn.Linear(24, 1)        
        
        
        en1_fc     = torch.Tensor(num_input, en1).fill_(0)            # 1995 -> 100
        en1_bias = torch.Tensor(en1).fill_(0)
        en2_fc     = torch.Tensor(en1, en2).fill_(0)             # 100  -> 100
        en2_bias = torch.Tensor(en2).fill_(0)
        mean_fc    = torch.Tensor(en2, num_topic).fill_(0)             # 100  -> 50
        mean_bias = torch.Tensor(num_topic).fill_(0)
        logvar_fc  = torch.Tensor(en2, num_topic).fill_(0)             # 100  -> 50
        logvar_bias = torch.Tensor(num_topic).fill_(0)
        
        self.register_buffer('en1_fc',    en1_fc)
        self.register_buffer('en2_fc',     en2_fc)
        self.register_buffer('mean_fc',  mean_fc) # buffers are just old storage, and are not updated with backpropagation.
        self.register_buffer('logvar_fc',     logvar_fc)
        self.register_buffer('en1_fc_bias',    en1_bias)
        self.register_buffer('en2_fc_bias',     en2_bias)
        self.register_buffer('mean_fc_bias',  mean_bias)
        self.register_buffer('logvar_fc_bias',     logvar_bias)
        
        
        self.en2_drop   = nn.Dropout(0.2)
        self.mean_bn    = nn.BatchNorm1d(num_topic)                      # bn for mean
        self.logvar_bn  = nn.BatchNorm1d(num_topic)                      # bn for logvar
        # z
        
        self.p_drop     = nn.Dropout(0.2)
        # decoder
        decoder = torch.Tensor(num_topic, num_input).fill_(0)   
        self.register_buffer('decoder', decoder)
        decoder_bias = torch.Tensor(num_input).fill_(0)
        self.register_buffer('decoder_bias', decoder_bias)

        self.decoder_bn = nn.BatchNorm1d(num_input)                      # bn for decoder
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, num_topic).fill_(0)
        prior_var    = torch.Tensor(1, num_topic).fill_(variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar) # buffers are just old storage, and are not updated with backpropagation.
        
        # initialize decoder weight
        # if init_mult != 0:
            #std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            # self.decoder.weight.data.uniform_(0, init_mult)
        # remove BN's scale parameters
        # self.logvar_bn.register_parameter('weight', None)
        # self.mean_bn.register_parameter('weight', None) # This causes errors because you can't have a none weight.
        # self.decoder_bn.register_parameter('weight', None)
        # self.decoder_bn.register_parameter('weight', None)

    def forward(self, input, compute_loss=False, avg_loss=True):
        # compute posterior
        inp_dim,en1,en2,num_topic,init_mult,variance,num_input = self.starterdict_parser(self.referraldict)

        en1_intermediate = self.hyp_en1_fc_1(input)
        en1_weights = self.hyp_en1_fc_2(en1_intermediate)
        en1_weights = F.sigmoid(en1_weights)

        en1_bias = self.hyp_en1_fc_b_1(en1_intermediate)
        en1_bias = self.hyp_en1_fc_b_1_dropout(en1_bias)
        en1_bias = self.hyp_en1_fc_b_2(en1_bias)
        en1_bias = F.sigmoid(en1_bias)
        
        
        en2_intermediate = self.hyp_en2_fc_1(en1_intermediate)
        en2_weights = self.hyp_en2_fc_2(en2_intermediate)
        en2_weights = F.sigmoid(en2_weights)
        en2_bias = self.hyp_en2_fc_b_1(en1_intermediate)
        en2_bias = self.hyp_en2_fc_b_1_dropout(en2_bias)
        en2_bias = self.hyp_en2_fc_b_2(en2_bias)
        en2_bias = F.sigmoid(en2_bias)
        
        
        mean_intermediate = self.hyp_mean_fc_1(en2_intermediate)
        mean_weights = self.hyp_mean_fc_2(mean_intermediate)
        mean_weights = F.sigmoid(mean_weights)
        mean_bias = self.hyp_mean_fc_b_1(mean_intermediate)
        mean_bias = self.hyp_mean_fc_b_1_dropout(mean_bias) # you might want to just use the same bias layer next time...
        mean_bias = self.hyp_mean_fc_b_2(mean_bias)
        mean_bias = F.sigmoid(mean_bias)



        logvar_intermediate = self.hyp_logvar_fc_1(en2_intermediate)
        logvar_weights = self.hyp_logvar_fc_2(logvar_intermediate)
        logvar_weights = F.sigmoid(logvar_weights)
        logvar_bias = self.hyp_logvar_fc_b_1(logvar_intermediate)
        logvar_bias = self.hyp_logvar_fc_b_1_dropout(logvar_bias)
        logvar_bias = self.hyp_logvar_fc_b_2(logvar_bias)
        logvar_bias = F.sigmoid(logvar_bias)
        decoder_intermediate = self.hyp_decoder_fc_1(torch.cat([mean_intermediate,logvar_intermediate],dim=1))
        decoder_weights = self.hyp_decoder_fc_2(decoder_intermediate)
        decoder_weights = F.sigmoid(decoder_weights) # might have to recurrent neural net this or something...
        
        decoder_bias = self.hyp_decoder_fc_b_1(decoder_intermediate)
        decoder_bias = self.hyp_decoder_fc_b_1_dropout(decoder_bias)
        decoder_bias = self.hyp_decoder_fc_b_2 (decoder_bias)
        
        self.en1_fc = en1_weights
        self.en2_fc = en2_weights
        self.mean_fc = mean_weights
        self.logvar_fc = logvar_weights
        self.decoder = decoder_weights
        self.en1_bias = en1_bias
        self.en2_bias = en2_bias
        self.mean_bias = mean_bias
        self.logvar_bias = logvar_bias
        self.decoder_bias = decoder_bias
        
        # print("suggested weight shape:",self.en1_fc.shape, self.en1_fc.reshape(-1,num_input,en1).shape)
        # print("input before and after: ",input.shape,input.reshape(input.shape[0],1,input.shape[1]).shape)
        # print("bias shape: ",en1_bias.shape)

        
        en1_fc_out = input.reshape(input.shape[0],1,input.shape[1]) @ self.en1_fc.view(-1,num_input,en1)
        # print("directly out of en1 shape:",en1_fc_out.shape)
        en1_fc_out = en1_fc_out.squeeze()
        en1_fc_out = en1_fc_out+en1_bias
        en1_fc_out = F.softplus(en1_fc_out)
        en1_fc_out = en1_fc_out.reshape(en1_fc_out.shape[0],1,en1_fc_out.shape[1])
        # print("en1_fc_out result:",en1_fc_out.shape)
        # print("target:", input.shape[0],en1)
        # print(en1_fc_out)
        # print("reshaped en2_fc:",self.en2_fc.reshape(-1,en1,en2).shape)
        # print("en2 fc bias shape:",en2_bias.shape)
        en2_fc_out =  en1_fc_out @ self.en2_fc.reshape(-1,en1,en2)
        # en2_fc_out =  torch.einsum("ij,ijj->ij",en1_fc_out.squeeze(),self.en2_fc.reshape(-1,en1,en2))
        # print("directly out of en2 shape:",en2_fc_out.shape)
        en2_fc_out = en2_fc_out.squeeze()
        en2_fc_out = en2_fc_out+en2_bias
        en2_fc_out = F.softplus(en2_fc_out)
        en2_dropped = self.en2_drop(en2_fc_out)
        # print("en2_fc_out reshaped shape: ",en2_dropped.reshape(en2_dropped.shape[0],1,en2_dropped.shape[1]).shape)
        # print("target:", input.shape[0],num_topic)
        
        # print("reshaped mean_fc:",self.mean_fc.reshape(-1,en2,num_topic).shape)
        # print("mean_fc bias shape:",mean_bias.shape)

                
        en2_dropped = en2_dropped.reshape(en2_dropped.shape[0],1,en2_dropped.shape[1])
        
        meanfc_out = en2_dropped @ self.mean_fc.reshape(-1,en2,num_topic)
        # print("directly out of mean_fc_out shape:",meanfc_out.shape)
        meanfc_out = meanfc_out.squeeze()
        meanfc_out = meanfc_out + self.mean_bias
        posterior_mean = self.mean_bn(meanfc_out)
        # print("mean Output Shape:",posterior_mean.shape)
        # print("target:", input.shape[0],num_topic)
        # print("reshaped logvar_fc:",self.logvar_fc.reshape(-1,en2,num_topic).shape)

        logvarfc_out = en2_dropped @ self.logvar_fc.reshape(-1,en2,num_topic) 
        # print("directly out of logvarfc_out shape:", logvarfc_out.shape)
        logvarfc_out = logvarfc_out.squeeze()

        logvarfc_out = logvarfc_out + self.logvar_bias
        logvarfc_out = F.sigmoid(logvarfc_out)
        posterior_logvar = self.logvar_bn(logvarfc_out)
        
        # print("posterior logvar Output Shape:",posterior_logvar.shape)
        # print("target:", input.shape[0],num_topic)
        # print(posterior_logvar)
        # print(posterior_mean)
        posterior_var = posterior_logvar.exp()
        # print(posterior_mean)
        # print(posterior_var.sqrt())
        
        
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z)                                                # mixture probability
        # print(z)
        p = self.p_drop(p)
        # print("reshaped decoder_fc:",self.decoder.reshape(-1,num_topic, num_input).shape)
        # print("p reshaped:",p.reshape(p.shape[0],-1,p.shape[1]).shape)
        # do reconstruction
        decoder_output = p.reshape(p.shape[0],-1,p.shape[1]) @ self.decoder.reshape(-1,num_topic, num_input)
        # print("directly out of decoder shape:", decoder_output.shape)
        decoder_output = decoder_output.squeeze()
        decoder_output = self.decoder_bias + decoder_output
        # print(decoder_output.shape)
        recon = F.softmax(decoder_output)             # reconstructed distribution over vocabulary
        # print("Decoder Output Shape:",recon.shape)
        # print("target:", input.shape[0],num_topic)

        if compute_loss:
            loss = self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
            # print(loss)
            return recon, loss , p
        else:
            return recon,p

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        # print("-"*50)
        # print(input)
        # print(recon)
        # print(posterior_mean)
        # print(posterior_logvar)
        # print(posterior_var)
        # print("-"*50)
        # input()
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean).expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var).expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar).expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.num_topic )
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean().reshape((-1))
        else:
            return loss

    def starterdict_parser(self,inp_dict):
        if "inp_dim" in inp_dict:
            inp_dim = inp_dict["inp_dim"]
        else:
            inp_dim = 1995
        if "en1" in inp_dict:
            en1 = inp_dict["en1"]
        else:
            en1 = 100
        if "en2" in inp_dict:
            en2 = inp_dict["en2"]
        else:
            en2 = 100
        
        if "num_topic" in inp_dict:
            num_topic = inp_dict["num_topic"]
        else:
            num_topic = 50
        
        if "init_mult" in inp_dict:
            init_mult = inp_dict['init_mult']
        else:
            init_mult = 1.0
            
        if "variance" in inp_dict:
            variance = inp_dict["variance"]
        else:
            variance = 0.995
        if "num_input" in inp_dict:
            num_input = inp_dict["num_input"]
        else:
            raise ValueError("Input dict did not specify starting layer's number of inputs. Can't determine shape.")
            
        return inp_dim,en1,en2,num_topic,init_mult,variance, num_input
        
        
class Hypernet_LDA_RNN(nn.Module):

    def __init__(self, inp_dict,device):
        super(Hypernet_LDA_RNN, self).__init__()
        inp_dim,en1,en2,num_topic,init_mult,variance,num_input = self.starterdict_parser(inp_dict)
        # encoder
        # num_input is misnamed and technically refers to the input vocabulary size, or the target for decoders to map to.
        
        self.device = device
        self.referraldict = inp_dict
        self.en1_fc     = nn.Linear(num_input, en1)             # 1995 -> 100
        self.en2_fc     = nn.Linear(en1, en2)             # 100  -> 100
        self.en2_drop   = nn.Dropout(0.2)
        self.decoder_bn = nn.BatchNorm1d(num_input)                      # bn for decoder

        
        self.meanRNN = torch.nn.RNN(input_size=num_input+en2+1,hidden_size=en2, num_layers=2, batch_first=True, bidirectional=False,dropout=0.1)
        self.mean_bias1 = torch.nn.Linear(en2,32)
        self.mean_bias2 = torch.nn.Linear(32,1)
        # same as below.
        # bias is added accordingly.
        
        
        self.logvarRNN = torch.nn.RNN(input_size=num_input+en2+1,hidden_size=en2, num_layers=2, batch_first=True, bidirectional=False,dropout=0.1)
        self.logvar_bias1 = torch.nn.Linear(en2,32)
        self.logvar_bias2 = torch.nn.Linear(32,1)
        # runs num_topic number of times also, but the output values are num_topic x en2.
        # then you need to remap it to en2, num_topic, as though it's a linear.
        
        
        self.decoderRNN = torch.nn.RNN(input_size=en2+1,hidden_size=256, num_layers=1, batch_first=True, bidirectional=False)
        self.decoder_bias1 = torch.nn.Linear(en2,64)
        self.decoder_bias2 = torch.nn.Linear(64,num_input)
        # runs num_topic number of times, where num topic is an input.
        self.decoderremapper = torch.nn.Linear(256,num_input)
        # remap to input number. Else it'll be massive.
        
        
        # z
        self.p_drop = nn.Dropout(0.2)
        self.en2_drop = nn.Dropout(0.2)
        # decoder
        # prior mean and variance as constant buffers
        prior_mean   = torch.Tensor(1, num_topic).fill_(0)
        prior_var    = torch.Tensor(1, num_topic).fill_(variance)
        prior_logvar = prior_var.log()
        self.register_buffer('prior_mean',    prior_mean)
        self.register_buffer('prior_var',     prior_var)
        self.register_buffer('prior_logvar',  prior_logvar) # buffers are just old storage, and are not updated with backpropagation.

        # initialize decoder weight
        # if init_mult != 0:
            #std = 1. / math.sqrt( ac.init_mult * (ac.num_topic + ac.num_input))
            # self.decoder.weight.data.uniform_(0, init_mult)
        # remove BN's scale parameters
        # self.logvar_bn.register_parameter('weight', None)
        # self.mean_bn.register_parameter('weight', None) # This causes errors because you can't have a none weight.
        # self.decoder_bn.register_parameter('weight', None)
        # self.decoder_bn.register_parameter('weight', None)

    def forward(self, input, num_topic, compute_loss=False, avg_loss=True):
        # compute posterior
        # print("Input Shape:",input.shape)
        inp_dim,en1,en2,_,init_mult,variance,num_input = self.starterdict_parser(self.referraldict)
        en1_out = F.softplus(self.en1_fc(input))                            # en1_fc   output
        en2_out = F.softplus(self.en2_fc(en1_out))                              # encoder2 output
        en2_out = self.en2_drop(en2_out)
        
        mean_bias1 = self.mean_bias1(en2_out)
        mean_bias = self.mean_bias2(mean_bias1)
        
        logvar_bias1 = self.logvar_bias1(en2_out)
        logvar_bias = self.logvar_bias2(logvar_bias1)
        
        decoder_bias1 = self.decoder_bias1(en2_out)
        decoder_bias = self.decoder_bias2(decoder_bias1)
        
        # print(input.shape)
        
        # print("en2 shape:",en2_out.shape)
        # print(torch.Tensor([[num_topic]]).tile((input.shape[0])).reshape(-1,1).shape)
        stacked_result = torch.cat([input,en2_out,torch.Tensor([[num_topic]]).tile((input.shape[0])).reshape(-1,1).to(self.device)],dim=1)
        decoder_input = torch.cat([en2_out,torch.Tensor([[num_topic]]).tile((input.shape[0])).reshape(-1,1).to(self.device)],dim=1)
        # print(stacked_result.shape)
        input_sequence = stacked_result.tile((num_topic)).reshape(-1,stacked_result.shape[0],stacked_result.shape[1])
        # print(input_sequence.shape)
        meanRNN_out, _ = self.meanRNN(input_sequence)
        
        
        logvarRNN_out, _ = self.logvarRNN(input_sequence)
        # runs num_topic number of times also, but the output values are num_topic x en2.
        # then you need to remap it to en2, num_topic, as though it's a linear.
        
        
        # print("Decoder input tiled shape:",decoder_input.tile((num_topic)).reshape(-1,decoder_input.shape[0],decoder_input.shape[1]).shape)
        decoderRNN_out, _ = self.decoderRNN(decoder_input.tile((num_topic)).reshape(-1,decoder_input.shape[0],decoder_input.shape[1]))
        mean_bn    = nn.BatchNorm1d(num_topic).to(self.device)                      # bn for mean
        logvar_bn  = nn.BatchNorm1d(num_topic).to(self.device)                      # bn for logvar
        decoder_bn = nn.BatchNorm1d(num_topic).to(self.device)                      # bn for decoder
        
        self.num_topic = num_topic
        # print(decoderRNN_out.reshape(-1, num_topic,128).shape)
        # print("original decoder unmapped out shape:",decoderRNN_out.shape)
        decoderRNN_out = self.decoderremapper(decoderRNN_out) 
        # needs to be reupdated according to how it's formulated in init
        
        
        # print(decoderRNN_out.shape)
        
        self.decoderRNN_out_latest = decoderRNN_out # Save decoder weights and num topics.
        
        # print("meanRNN out reshaped:",meanRNN_out.reshape(input.shape[0],en2,num_topic).shape)
        # print("en2 Linear out shape:",en2_out.reshape(en2_out.shape[0],1,en2_out.shape[1]).shape)
        meanfc_out = en2_out.reshape(en2_out.shape[0],1,en2_out.shape[1]) @ meanRNN_out.reshape(-1,en2,num_topic)
        # print("directly out of mean_fc_out shape:",meanfc_out.shape)
        meanfc_out = meanfc_out.squeeze()
        meanfc_out = meanfc_out + mean_bias
        # print("meanfc_out shape:",meanfc_out.shape)
        posterior_mean = mean_bn(meanfc_out)
        # print("mean Output Shape:",posterior_mean.shape)
        # print("target:", input.shape[0],num_topic)
        
        # print("reshaped logvarRNN_out:",logvarRNN_out.reshape(-1,en2,num_topic).shape)
        logvarfc_out = en2_out.reshape(en2_out.shape[0],1,en2_out.shape[1]) @ logvarRNN_out.reshape(-1,en2,num_topic) 
        # print("directly out of logvarfc_out shape:", logvarfc_out.shape)
        logvarfc_out = logvarfc_out.squeeze()

        logvarfc_out = logvarfc_out + logvar_bias
        logvarfc_out = F.sigmoid(logvarfc_out)
        posterior_logvar = logvar_bn(logvarfc_out)
        
        # print("posterior logvar Output Shape:",posterior_logvar.shape)
        # print("target:", input.shape[0],num_topic)
        # print(posterior_logvar)
        # print(posterior_mean)
        posterior_var = posterior_logvar.exp()
        # print(posterior_mean)
        # print(posterior_var.sqrt())
        
        
        # take sample
        eps = Variable(input.data.new().resize_as_(posterior_mean.data).normal_()) # noise
        z = posterior_mean + posterior_var.sqrt() * eps                 # reparameterization
        p = F.softmax(z)                                                # mixture probability
        # print(z)
        p = self.p_drop(p)
        # print("reshaped decoder_fc:",decoderRNN_out.reshape(-1,num_topic, num_input).shape)
        # print("p reshaped:",p.reshape(p.shape[0],-1,p.shape[1]).shape)
        # do reconstruction
        decoder_output =  p.reshape(p.shape[0],-1,p.shape[1]) @ decoderRNN_out.reshape(-1, num_topic,num_input)
        # print("directly out of decoder shape:", decoder_output.shape)
        # print(decoder_bias.shape)
        decoder_output = decoder_output.squeeze()
        
        decoder_output = decoder_bias + decoder_output
        # print(decoder_output.shape)
        recon = F.softmax(decoder_output.squeeze(),dim=1)             # reconstructed distribution over vocabulary
        # print("Decoder Output Shape:",recon.shape)
        # print("target:", input.shape[0],num_input)

        if compute_loss:
            loss = self.loss(input, recon, posterior_mean, posterior_logvar, posterior_var, avg_loss)
            # print(loss)
            return recon, loss , p
        else:
            return recon,p

    def loss(self, input, recon, posterior_mean, posterior_logvar, posterior_var, avg=True):
        # NL
        # print("-"*50)
        # print(input)
        # print(recon)
        # print(posterior_mean)
        # print(posterior_logvar)
        # print(posterior_var)
        # print("-"*50)
        # input()
        NL  = -(input * (recon+1e-10).log()).sum(1)
        # KLD, see Section 3.3 of Akash Srivastava and Charles Sutton, 2017, 
        # https://arxiv.org/pdf/1703.01488.pdf
        prior_mean   = Variable(self.prior_mean)[:,posterior_mean.shape[1]-1].expand_as(posterior_mean)
        prior_var    = Variable(self.prior_var)[:,posterior_mean.shape[1]-1].expand_as(posterior_mean)
        prior_logvar = Variable(self.prior_logvar)[:,posterior_mean.shape[1]-1].expand_as(posterior_mean)
        var_division    = posterior_var  / prior_var
        diff            = posterior_mean - prior_mean
        diff_term       = diff * diff / prior_var
        logvar_division = prior_logvar - posterior_logvar
        # put KLD together
        KLD = 0.5 * ( (var_division + diff_term + logvar_division).sum(1) - self.num_topic )
        # loss
        loss = (NL + KLD)
        # in traiming mode, return averaged loss. In testing mode, return individual loss
        if avg:
            return loss.mean().reshape((-1))
        else:
            return loss

    def starterdict_parser(self,inp_dict):
        if "inp_dim" in inp_dict:
            inp_dim = inp_dict["inp_dim"]
        else:
            inp_dim = 1995
        if "en1" in inp_dict:
            en1 = inp_dict["en1"]
        else:
            en1 = 100
        if "en2" in inp_dict:
            en2 = inp_dict["en2"]
        else:
            en2 = 100
        
        if "num_topic" in inp_dict:
            num_topic = inp_dict["num_topic"]
        else:
            num_topic = 50
        
        if "init_mult" in inp_dict:
            init_mult = inp_dict['init_mult']
        else:
            init_mult = 1.0
            
        if "variance" in inp_dict:
            variance = inp_dict["variance"]
        else:
            variance = 0.995
        if "num_input" in inp_dict:
            num_input = inp_dict["num_input"]
        else:
            raise ValueError("Input dict did not specify starting layer's number of inputs. Can't determine shape.")
            
        return inp_dim,en1,en2,num_topic,init_mult,variance, num_input