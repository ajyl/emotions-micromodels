import transformers
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch

from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput

# For OrderRecovery
from allrank.models.losses import listMLE, listNet

# To use as the model for DialogMLM
from transformers import BertForMaskedLM

import torch.nn.functional as F

import spacy
#from other_models import Similarity

import transformers
import torch.nn as nn

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

    def temp_forward(self, x, y, temp):
        return self.cos(x, y) / temp


##############################
##############################
##############################
##############################
class CrossScorer(nn.Module):
    """
    Note: This is the bi encoder (separate) model.
          TODO: Fix name accordingly
    """

    def __init__(self, p_encoder, r_encoder, use_aux_loss=False): #, tokenizer):
        """

        """
        super(CrossScorer, self).__init__()

        self.p_encoder = p_encoder
        self.r_encoder = r_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        #self.sim = nn.CosineSimilarity(dim=-1)
        self.temp = 0.1
        self.sim = Similarity(temp=self.temp)

        self.hn_weight = 1.0

        self.lamb_1 = 0.5

        self.freeze_response = False
        if self.freeze_response:
            for param in self.r_encoder.parameters():
                param.requires_grad = False
 
        #self.tokenizer = tokenizer
        #self.nlp = spacy.load('en_core_web_sm')
        #self.nlp.tokenizer.rules = {} #{key: value for key, value in self.nlp.tokenizer.rules.items() if "'" not in key and "’" not in key and "‘" not in key and "gon" not in key and "can" not in key and "got" not in key}

        self.use_aux_loss = use_aux_loss
        self.encoder_type = "bi"

    def get_pos_embeds(self, pos_masks, last_hidden):
        #pos_masks = pos_masks.to(self.device)

        pos_embeds = pos_masks.unsqueeze(-1) * last_hidden
        #print(pos_embeds.size())
        pos_embeds = torch.sum(pos_embeds, dim=1)
        return pos_embeds



    def score_forward(
        self,
        p_batch=None,
        r_batch=None,
        pv=None,
        pn=None,
        rv=None,
        rn=None,
        ):  

        """
        TODO: decide where to prepare the strings into the batch and masks
              and do demo
        """

        p_output, verb_p, noun_p = self.p_encoder.emb_forward(
                **p_batch, verb_mask=pv, noun_mask=pn
                )        

        r_output, verb_r, noun_r = self.r_encoder.emb_forward(
                **r_batch, verb_mask =rv, noun_mask=rn
                )       
        #if hn_batch:
        #    hn_output = self.r_encoder.emb_forward(
        #            **hn_batch
        #            )    

        #p_full = p_output.last_hidden_state
        #r_full = r_output.last_hidden_state

        p_z = p_output #p_output.last_hidden_state[:,0,:]
        r_z = r_output #r_output.last_hidden_state[:,0,:]

        #p_z = F.normalize(p_z, dim=1)
        #r_z = F.normalize(r_z, dim=1)
 
        cos_sim, cos_denom = self.flat_sim(p_z, r_z)
        #"""
        if self.use_aux_loss:
            #noun_p = self.get_pos_embeds(pn, p_full)
            #noun_r = self.get_pos_embeds(rn, r_full)
            #verb_p = self.get_pos_embeds(pv, p_full)
            #verb_r = self.get_pos_embeds(rv, r_full)


            noun_sim, denom_n = self.flat_sim(noun_p,noun_r)
            verb_sim, denom_v = self.flat_sim(verb_p,verb_r)
            
           
            # TODO: normalize with appropriate denominator 

            #"""
            cos_sim = cos_sim #/ p_batch["input_ids"].size(0)
            verb_sim = verb_sim #/ denom_v
            noun_sim = noun_sim #/ denom_n
            score = cos_sim + verb_sim + noun_sim
            # normalizing by 3 (uniform weight)
            score = score /3.0
            #"""
            #combined_p = torch.cat((p_z, noun_p, verb_p), -1)
            #combined_r = torch.cat((r_z, noun_r, verb_r), -1)
            #score, denom = self.flat_sim(combined_p, combined_r)
        else:
            score = cos_sim
        #"""
        #score = cos_sim
        
        return score

    def flat_sim(self, z1, z2):
        return self.sim(z1,z2), z1.size(0)


    def cl_loss(self, z1, z2, hn_batch=None, hn_output=None):

        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        
        # (6) Get the diagonal truth labels and compute loss
        #     labels: B x W
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)

        ce_loss_fct = nn.CrossEntropyLoss()
        ce_loss = ce_loss_fct(cos_sim, labels)
        

        """
        TODO: following part is for hard negatives
        """
        if hn_batch:
            # prompt is used here (prompt / z1)
            hn_z = hn_output.last_hidden_state[:,0,:]
            p_hn_cos = self.sim(z_1.unsqueeze(1), hn_z.unsqueeze(0))
            cos_sim = torch.cat([cos_sim,p_hn_cos],1)

        # (6) Get the diagonal truth labels and compute loss
        #     labels: B
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        
        if hn_batch:
            # TODO: here needs to be to(accelerator.device)
            weights = torch.tensor([[0.0] * (cos_sim.size(-1) - p_hn_cos.size(-1)) + [0.0] * i + [self.hn_weight] + [0.0] * (p_hn_cos.size(-1) - i - 1) for i in range(p_hn_cos.size(-1))]).to(self.device)
            cos_sim = cos_sim + weights
        
        
        return ce_loss, cos_sim



    def forward(
        self,
        p_batch=None,
        r_batch=None,
        pv=None,
        pn=None,
        rv=None,
        rn=None,
        #verb_mask=None,
        #noun_mask=None,
        hn_batch=None
        ):  

        
        p_output, verb_p, noun_p = self.p_encoder.emb_forward(
                **p_batch, verb_mask=pv, noun_mask=pn
                )        

        r_output, verb_r, noun_r = self.r_encoder.emb_forward(
                **r_batch, verb_mask =rv, noun_mask=rn
                )        
        if hn_batch:
            hn_output = self.r_encoder.emb_forward(
                    **hn_batch
                    )    

        #p_full = p_output.last_hidden_state
        #r_full = r_output.last_hidden_state

        p_z = p_output#p_output.last_hidden_state[:,0,:]
        r_z = r_output #r_output.last_hidden_state[:,0,:]

        #p_z = F.normalize(p_z, dim=1)
        #r_z = F.normalize(r_z, dim=1)
 
        ce_loss, cos_sim = self.cl_loss(p_z, r_z, hn_batch, None)
            
        if self.use_aux_loss:
            """
            noun_p = self.get_pos_embeds(pn, p_full)
            noun_r = self.get_pos_embeds(rn, r_full)
            verb_p = self.get_pos_embeds(pv, p_full)
            verb_r = self.get_pos_embeds(rv, r_full)
            """

            noun_loss, noun_sim = self.cl_loss(noun_p,noun_r)
            verb_loss, verb_sim = self.cl_loss(verb_p,verb_r)
            loss = ce_loss + verb_loss + noun_loss #+ verb_loss

            #combined_p = torch.cat((p_z, noun_p, verb_p), -1)
            #combined_r = torch.cat((r_z, noun_r, verb_r), -1)
            #loss, sim= self.cl_loss(combined_p, combined_r)

        else:
            loss = ce_loss
        #mse_loss_fct = nn.MSELoss()
        #mse_loss = mse_loss_fct(z1, z2)
        #loss = self.lamb_1 * mse_loss + (1-self.lamb_1) *ce_loss
        #score = cos_sim + verb_sim + noun_sim
        
        #loss = ce_loss
        #score = cos_sim
        return SequenceClassifierOutput(
                loss=loss,
                #logits=score,
                #hidden_states=bert_output.hidden_states,
                #attentions=bert_output.attentions,
                )
 
    def cl_loss_with_hn(self, prompts, responses):
        """
        1 prompt corresponds to multiple responses
        the first response is always the good one
        the remaining responses are bad (hard negatives)
        """
        BSZ = prompts.size(0)
        # prompt 2, Hdim
        # responses 8, Hdim
        responses = list(responses.tensor_split(BSZ, dim=0) )#[ responses   ]
        # responses 2, 4, Hdim
        responses = torch.stack(responses)
        # 2 != 4
        assert prompts.size(0) == responses.size(0)
        ce_loss_fct = nn.CrossEntropyLoss()

        loss = 0
        sim = 0
        for prompt, response_set in zip(prompts, responses):
            # prompt: Hdim
            # responset_set: 4, Hdim
            # cos_sim should be (4, )
            cos_sim = self.sim(prompt.unsqueeze(0), response_set)
            # always the first response is the good response
            label = torch.LongTensor([0]).to(self.device) #torch.arange(cos_sim.size(0)).long().to(self.device)
            #print("cos_sim:", cos_sim.size())
            #exit()
            cos_sim = cos_sim.unsqueeze(0)
            ce_loss = ce_loss_fct(cos_sim, label)
            loss += ce_loss
            sim += cos_sim
           
        return loss, sim


       
    def hard_forward(
        self,
        p_batch=None,
        r_batches=None,
        pv=None,
        pn=None,
        rv=None,
        rn=None,
        ):  



        p_output, verb_p, noun_p = self.p_encoder.emb_forward(
                **p_batch, verb_mask=pv, noun_mask=pn
                )        

        r_outputs, verb_rs, noun_rs = self.r_encoder.emb_forward(
                **r_batches, verb_mask =rv, noun_mask=rn
                )        

        #p_full = p_output.last_hidden_state
        #r_fulls = r_outputs.last_hidden_state
        
        p_z = p_output #p_output.last_hidden_state[:,0,:]
        r_zs = r_outputs #r_outputs.last_hidden_state[:,0,:]

        #p_z = F.normalize(p_z, dim=1)
        #r_z = F.normalize(r_z, dim=1)
        
        ce_loss, cos_sim = self.cl_loss_with_hn(p_z, r_zs)
            
        if self.use_aux_loss:
            """
            noun_p = self.get_pos_embeds(pn, p_full)
            noun_rs = self.get_pos_embeds(rn, r_fulls)
            verb_p = self.get_pos_embeds(pv, p_full)
            verb_rs = self.get_pos_embeds(rv, r_fulls)
            """

            noun_loss, noun_sim = self.cl_loss_with_hn(noun_p,noun_rs)
            verb_loss, verb_sim = self.cl_loss_with_hn(verb_p,verb_rs)
            loss = ce_loss + verb_loss + noun_loss 
    
            #combined_p = torch.cat((p_z, noun_p, verb_p), -1)
            #combined_rs = torch.cat((r_zs, noun_rs, verb_rs), -1)
            #loss, sim= self.cl_loss_with_hn(combined_p, combined_rs)
        else:
            loss = ce_loss
        #mse_loss_fct = nn.MSELoss()
        #mse_loss = mse_loss_fct(z1, z2)
        #loss = self.lamb_1 * mse_loss + (1-self.lamb_1) *ce_loss
        #score = cos_sim + verb_sim + noun_sim
        
        #loss = ce_loss
        #score = cos_sim
        return SequenceClassifierOutput(
                loss=loss,
                #logits=score,
                #hidden_states=bert_output.hidden_states,
                #attentions=bert_output.attentions,
                )
        

##############################
##############################
##############################
##############################
class CrossScorerCrossEncoder(nn.Module):
    """
    TODO: Rethink how the CL loss is computed
          For now, for Cross Encoder which generaes scalar relevance score
          Triplet Ranking Loss makes sense?

    Scorer model with Cross Encoder + Binary Prediction Head
    Note that here we do not do POS augmentation
         since it does not make sense under this setting

    """
    def __init__(self, transformer): #, tokenizer):
        """

        """
        super(CrossScorerCrossEncoder, self).__init__()

        self.cross_encoder = transformer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        #self.temp = 0.1
        #self.sim = Similarity(temp=self.temp)

        
        # Binary Head
        self.l1 = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,1)
        # 3 = T, C + Special Token
        #self.l3 = torch.nn.Linear(256,1)
        # Then use this with BCELossWithLogits
       
        # TODO: make sure that EOT is added to vocab
        # Man just use a separator... (why not?)
        # It's not like sentence units are too meaningful...?
        self.encoder_type = "cross"    

        # TODO: remove this!!!!
        #self.l2_classify = torch.nn.Linear(512,3)

    def saved_score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):  
        """
        DO NOT FORGET to remove this and the forward func 
        and revert the saved_ones
        """


        output = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pair_reps = output.last_hidden_state[:,0,:]
        logits = self.l2_classify(self.relu(self.l1(pair_reps)))
        
        #score = score.sigmoid()
        return logits


    def score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_attentions=False
        ):  


        output = self.cross_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pair_reps = output.last_hidden_state[:,0,:]
        score = self.l2(self.relu(self.l1(pair_reps)))
        
        if output_attentions and return_attentions:
            return score.sigmoid().squeeze(), output.attentions

        return score

    def cl_loss_all_random(self, pair_scores, labels):
        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        BSZ = int(BSZ/4)

        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        
        
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
              
        lq_scores = pair_scores[:,1:] # 3

        # Use torch.clone to match Positive to Negatives
        hq_scores = pair_scores[:,0] #.repeat(1,neg_scores.size(-1)).flatten()
        # 6
        #target = torch.ones(pos_scores.size()).to(self.device)
        

        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        
        loss = hq_lq_loss
        return loss



    def cl_loss(self, pair_scores, labels):
        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        BSZ = int(BSZ/(4))

        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        
        
        gap_1_loss_fct = nn.MarginRankingLoss(margin=0.5)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
              
        mq_scores = pair_scores[:,1] # 1
        lq_scores = pair_scores[:,2:-1] # 2

        # Use torch.clone to match Positive to Negatives
        hq_scores = pair_scores[:,0] #.repeat(1,neg_scores.size(-1)).flatten()
        # 6
        #target = torch.ones(pos_scores.size()).to(self.device)
        
        hq_mq_loss = gap_1_loss_fct(
                hq_scores.flatten(), 
                mq_scores.flatten(), 
                torch.ones(mq_scores.flatten().size()).to(self.device))
        mq_lq_loss = gap_1_loss_fct(
                mq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        
        mismatch_scores = pair_scores[:,-1]
        hq_mismatch_loss =  gap_2_loss_fct(
                        hq_scores.flatten(), 
                        mismatch_scores.flatten(), 
                        torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mq_mismatch_loss = gap_1_loss_fct(
                mq_scores.flatten(), 
                mismatch_scores.flatten(), 
                torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mismatch_loss = hq_mismatch_loss + mq_mismatch_loss 

        # NOTE: comment out the mismatch loss for ablation of prompt-aware loss
        loss = hq_mq_loss + mq_lq_loss + hq_lq_loss  + mismatch_loss 
        return loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        random = False
        ):
 
        pair_scores = self.score_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).sigmoid().squeeze()

        if True:
            loss_fct = torch.nn.MSELoss()

            BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
            #print("size(0)", BSZ)
            BSZ = int(BSZ/(4+1))
            #print("bsz", BSZ)

    
            label = torch.zeros(5).float() #.float() 
            label[0] = 1.0
            label[1] = 0.5
            labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
    

            if True:
                """
                # remove every 5th item from pair_scores
                # THIS IS FOR ABLATION STUDY OF naive_regression prompt loss
                # BECAUSE 5th item is the mismatch score
                """

                # BSZ == 2 or BSZ == 4
                
                import numpy as np
                idx = np.array([i for i in range(len(pair_scores)) if i%5!=4])
                pair_scores = pair_scores[idx]
                labels = labels[idx]
                # print(pair_scores.size())
                # print(labels.size())
                # print()

            #labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
            """
            here taking care of prompt-aware terms for ablation
            """

   
            reg_loss = loss_fct(pair_scores, labels)
            return SequenceClassifierOutput(
                loss=reg_loss,
                logits=pair_scores,
                )


        # BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        # BSZ = int(BSZ/4)

        # label = torch.zeros(4).long()
        # label[0] = 1
        # labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
        labels = None
        # 2 != 4

        #pred_loss_fct = torch.nn.BCEWithLogitsLoss()

        #pred_loss = pred_loss_fct(pair_scores, labels)
        
        # TODO: For both loss functions,
        #       Add Prompt-Switch Loss 
        if not random:
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            cl_loss = self.cl_loss_all_random(pair_scores, labels)
   

        loss =   cl_loss #+ reg_loss
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
 
    def saved_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        random = False
        ):
        """
        temporarily using this for regression baseline
        revert saved_forward to forward and remove this
        """
 
        pair_scores = self.score_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).squeeze()

        #loss_fct = torch.nn.CrossEntropyLoss()
        loss_fct = torch.nn.MSELoss()

        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        #print("size(0)", BSZ)
        BSZ = int(BSZ/(4+1))
        #print("bsz", BSZ)

 
        label = torch.zeros(5).float() #.float() 
        label[0] = 1.0
        label[1] = 0.5
        labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
   

        if False:
            """
            # remove every 5th item from pair_scores
            # THIS IS FOR ABLATION STUDY OF naive_regression prompt loss
            """

            # BSZ == 2 or BSZ == 4
            
            import numpy as np
            idx = np.array([i for i in range(len(pair_scores)) if i%5!=4])
            pair_scores = pair_scores[idx]
            labels = labels[idx]
            # print(pair_scores.size())
            # print(labels.size())
            # print()

        #labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
        """
        here taking care of prompt-aware terms for ablation
        """

        # 2 != 4
            
        #print(pair_scores.size())
        #print(labels.size())
        reg_loss = loss_fct(pair_scores, labels)
        return SequenceClassifierOutput(
                loss=reg_loss,
                logits=pair_scores,
                )
         #pred_loss_fct = torch.nn.BCEWithLogitsLoss()

        #pred_loss = pred_loss_fct(pair_scores, labels)
        
        # TODO: For both loss functions,
        #       Add Prompt-Switch Loss 
        if not random:
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            cl_loss = self.cl_loss_all_random(pair_scores, labels)
   

        loss =  cl_loss # + pred_loss
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
 
          
    def hard_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        return self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )



#############################################################
#############################################################
#############################################################

class CrossScorerBiEncoder(nn.Module):
    """
    TODO: Rethink how the CL loss is computed
          For now, for Cross Encoder which generaes scalar relevance score
          Triplet Ranking Loss makes sense?

    Scorer model with Cross Encoder + Binary Prediction Head
    Note that here we do not do POS augmentation
         since it does not make sense under this setting

    """
    def __init__(self): #, tokenizer):
        """

        """
        super(CrossScorerBiEncoder, self).__init__()

        self.p_encoder = AutoModel.from_pretrained("roberta-base")
        self.r_encoder = AutoModel.from_pretrained("roberta-base")

        self.attention = nn.MultiheadAttention(768, 1)

        self.l1 = torch.nn.Linear(768, 512)
        self.relu = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,1)


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        #self.sim = nn.CosineSimilarity(dim=-1)
        self.temp = 0.1
        self.sim = Similarity(temp=self.temp)

        self.encoder_type = "bi"    

    def score_forward(
        self,
        p_batch=None,
        r_batch=None
        ):  

        p_output = self.p_encoder(
                **p_batch
        )

        r_output = self.r_encoder(
                **r_batch
        )        
        p_pooled = p_output.last_hidden_state[:,0,:].unsqueeze(0)#.transpose(1,0)
        r_hiddens = r_output.last_hidden_state.transpose(1,0)
        
        #print(p_pooled.size())
        #print(r_hiddens.size())

        #pair_reps = output.last_hidden_state[:,0,:]
        attn_output, attn_output_weights = self.attention(p_pooled, r_hiddens, r_hiddens)
        attn_output = attn_output.transpose(1,0)
        #print(attn_output.size())
        #print(attn_output_weights.size())

        score = self.l2(self.relu(self.l1(attn_output)))
        
        return score

    def cl_loss_all_random(self, pair_scores, labels):
        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        BSZ = int(BSZ/4)

        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        
        
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
              
        lq_scores = pair_scores[:,1:] # 3

        # Use torch.clone to match Positive to Negatives
        hq_scores = pair_scores[:,0] #.repeat(1,neg_scores.size(-1)).flatten()
        # 6
        #target = torch.ones(pos_scores.size()).to(self.device)
        

        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        
        loss = hq_lq_loss
        return loss



    def cl_loss(self, pair_scores, labels):
        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        BSZ = int(BSZ/4)

        pair_scores= list(pair_scores.tensor_split(BSZ, dim=0) )
        pair_scores = torch.stack(pair_scores)
        
        
        gap_1_loss_fct = nn.MarginRankingLoss(margin=0.5)
        gap_2_loss_fct = nn.MarginRankingLoss(margin=1.0)
              
        mq_scores = pair_scores[:,1] # 1
        lq_scores = pair_scores[:,2:-1] # 2

        # Use torch.clone to match Positive to Negatives
        hq_scores = pair_scores[:,0] #.repeat(1,neg_scores.size(-1)).flatten()
        # 6
        #target = torch.ones(pos_scores.size()).to(self.device)
        
        hq_mq_loss = gap_1_loss_fct(
                hq_scores.flatten(), 
                mq_scores.flatten(), 
                torch.ones(mq_scores.flatten().size()).to(self.device))
        mq_lq_loss = gap_1_loss_fct(
                mq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        hq_lq_loss = gap_2_loss_fct(
                hq_scores.repeat(1,lq_scores.size(-1)).flatten(), 
                lq_scores.flatten(), 
                torch.ones(lq_scores.flatten().size()).to(self.device))
        
        mismatch_scores = pair_scores[:,-1]
        hq_mismatch_loss =  gap_2_loss_fct(
                        hq_scores.flatten(), 
                        mismatch_scores.flatten(), 
                        torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mq_mismatch_loss = gap_1_loss_fct(
                mq_scores.flatten(), 
                mismatch_scores.flatten(), 
                torch.ones(mismatch_scores.flatten().size()).to(self.device))
        mismatch_loss = hq_mismatch_loss + mq_mismatch_loss 
    
        loss = hq_mq_loss + mq_lq_loss + hq_lq_loss + mismatch_loss 
        return loss


    def forward(
        self,
        p_batch = None,
        r_batch = None,
        random = False
        ):
        """
        for just regression 
        """
 
        pair_scores = self.score_forward(
                p_batch, r_batch
        ).squeeze()

    
        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        BSZ = int(BSZ/4)

        label = torch.zeros(4).long()
        label[0] = 1
        labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
        # 2 != 4

        #pred_loss_fct = torch.nn.BCEWithLogitsLoss()

        #pred_loss = pred_loss_fct(pair_scores, labels)
        
        # TODO: For both loss functions,
        #       Add Prompt-Switch Loss 
        if not random:
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            cl_loss = self.cl_loss_all_random(pair_scores, labels)
   

        loss =  cl_loss # + pred_loss
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
 
 

    def saved_forward(
        self,
        p_batch = None,
        r_batch = None,
        random = False
        ):
 
        pair_scores = self.score_forward(
                p_batch, r_batch
        ).squeeze()

    
        BSZ = pair_scores.size(0) # BSZ=2 * 4 (# Pos + # Neg) = 8 
        BSZ = int(BSZ/4)

        label = torch.zeros(4).long()
        label[0] = 1
        labels = torch.cat( [ label for x in range(BSZ)], -1).float().to(self.device)
        # 2 != 4

        #pred_loss_fct = torch.nn.BCEWithLogitsLoss()

        #pred_loss = pred_loss_fct(pair_scores, labels)
        
        # TODO: For both loss functions,
        #       Add Prompt-Switch Loss 
        if not random:
            cl_loss = self.cl_loss(pair_scores, labels)
        else:
            cl_loss = self.cl_loss_all_random(pair_scores, labels)
   

        loss =  cl_loss # + pred_loss
        return SequenceClassifierOutput(
                loss=loss,
                logits=pair_scores,
                )
 
          
    def hard_forward(
        self,
        p_batch=None,
        r_batch=None
        ):

        return self.forward(
                p_batch, r_batch
        )



###########################
##############################
##############################
##############################
class CrossScorerWithHead(nn.Module):
    """
    Not used
    """
    def __init__(self, p_encoder, r_encoder):
        """

        """
        super(CrossScorerWithHead, self).__init__()

        self.p_encoder = p_encoder
        self.r_encoder = r_encoder
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        #self.sim = nn.CosineSimilarity(dim=-1)
        #self.hn_weight = 1.0
        
        # Head
        self.l1 = torch.nn.Linear(768*3, 512)
        self.relu1 = torch.nn.ELU()
        self.l2 = torch.nn.Linear(512,256)
        self.relu2 = torch.nn.ELU()
        # 3 = T, C + Special Token
        self.l3 = torch.nn.Linear(256,3)
                
    
    def score_forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):

        return None

    def forward(
        self,
        p_batch=None,
        r_batch=None,
        labels = None
        ):  
        
        p_output = self.p_encoder.emb_forward(
                **p_batch
                )        

        r_output = self.r_encoder.emb_forward(
                **r_batch
                )       

        p_z = p_output.last_hidden_state[:,0,:]
        r_z = r_output.last_hidden_state[:,0,:]

        z = torch.cat([p_z,r_z,torch.abs(p_z-r_z)],dim=-1)       
        z = self.l3(self.relu2(self.l2(self.relu1(self.l1(z)))))
        if labels is None:
            return SequenceClassifierOutput(logits=z)
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(z, labels)
        
        
        return SequenceClassifierOutput(
                loss=loss,
                logits=z,
                #hidden_states=bert_output.hidden_states,
                #attentions=bert_output.attentions,
                )
        


