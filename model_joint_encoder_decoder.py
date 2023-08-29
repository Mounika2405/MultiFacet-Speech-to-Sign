# coding: utf-8
"""
Module to represents whole models
"""

import numpy as np
import torch.nn as nn
from torch import Tensor
import torch

from initialization import initialize_model
from embeddings import Embeddings
from encoders import Encoder, TransformerEncoder, MockEncoder #, WhisperEncoder
from decoders import Decoder, TransformerDecoder, FACSDecoder, MockDecoder, JointTransformerDecoder
from constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, TARGET_PAD
from search import greedy
from vocabulary import Vocabulary
from batch import Batch
from discriminator import Classifier, ClassifierLayers, MockClassifier


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 text_encoder: Encoder,
                 decoder: Decoder,
                 nonreg_decoder: Decoder,
                 src_embed: Embeddings,
                 text_src_embed: Embeddings,
                 trg_embed: Embeddings,
                 nonreg_trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 nonreg_trg_vocab: Vocabulary,
                 cfg: dict,
                 in_trg_size: int,
                 out_trg_size: int,
                 facs_embed: Embeddings,
                 facs_decoder: Decoder,
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        model_cfg = cfg["model"]

        self.src_embed = src_embed
        self.text_src_embed = text_src_embed
        self.trg_embed = trg_embed
        self.facs_trg_embed = facs_embed
        self.nonreg_trg_embed = nonreg_trg_embed
        

        self.encoder = encoder
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.nonreg_decoder = nonreg_decoder
        self.facs_decoder = facs_decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.nonreg_trg_vocab = nonreg_trg_vocab
        self.bos_index = 0 
        self.pad_index = 0.0
        self.eos_index = 0 
        self.target_pad = TARGET_PAD
        self.nonreg_bos_index = self.nonreg_trg_vocab.stoi[BOS_TOKEN]
        self.nonreg_pad_index = self.nonreg_trg_vocab.stoi[PAD_TOKEN]
        self.nonreg_eos_index = self.nonreg_trg_vocab.stoi[EOS_TOKEN]

        self.use_cuda = cfg["training"]["use_cuda"]
        self.batch_size = cfg["training"]["batch_size"]
        self.in_trg_size = in_trg_size
        self.out_trg_size = out_trg_size
        self.count_in = model_cfg.get("count_in",True)
        # Just Counter
        self.just_count_in = model_cfg.get("just_count_in",False)
        # Gaussian Noise
        self.gaussian_noise = model_cfg.get("gaussian_noise",False)
        # Gaussian Noise
        if self.gaussian_noise:
            self.noise_rate = model_cfg.get("noise_rate", 1.0)

        # Future Prediction - predict for this many frames in the future
        self.future_prediction = model_cfg.get("future_prediction", 0)

    # pylint: disable=arguments-differ
    def forward(self,
                src: Tensor,
                trg_input: Tensor,
                nonreg_trg_input: Tensor,
                src_mask: Tensor,
                src_lengths: Tensor,
                trg_mask: Tensor = None,
                nonreg_trg_mask: Tensor = None,
                src_input: Tensor = None,
                noise_rate:int=0,
                facs_trg_input: Tensor = None,
                facs_trg_mask: Tensor = None,
                text_src: Tensor = None,
                text_src_mask: Tensor = None,
                ) -> (
        Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """

        # Encode the source sequence
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)

        text_encoder_output, text_encoder_hidden = self.encode_text(text_src=text_src, 
                                                                    text_src_length=None, 
                                                                    text_src_mask=text_src_mask)

        unroll_steps = trg_input.size(1)

        # Add gaussian noise to the target inputs, if in training
        if (self.gaussian_noise) and (self.training) and (self.out_stds is not None):

            # Create a normal distribution of random numbers between 0-1
            noise = trg_input.data.new(trg_input.size()).normal_(0, 1)
            # Zero out the noise over the counter
            noise[:,:,-1] = torch.zeros_like(noise[:, :, -1])

            # Need to add a zero on the end of
            if self.future_prediction != 0:
                self.out_stds = torch.cat((self.out_stds,torch.zeros_like(self.out_stds)))[:trg_input.shape[-1]]

            # Need to multiply by the standard deviations
            noise = noise * self.out_stds

            # Add to trg_input multiplied by the noise rate
            trg_input = trg_input + noise_rate*noise

        # Decode the target pose sequence
        skel_out, dec_hidden, _, _ = self.decode(encoder_output=encoder_output,
                                                 text_encoder_output=text_encoder_output,
                                                 src_mask=src_mask, text_src_mask=text_src_mask,
                                                 trg_input=trg_input, trg_mask=trg_mask)
        # Decode the target text sequence
        nonreg_out, dec_h, _, _ = self.nonreg_decode(encoder_output=encoder_output,
                                                 src_mask=src_mask, nonreg_trg_input=nonreg_trg_input,
                                                 nonreg_trg_mask=nonreg_trg_mask)
        gloss_out = None
        facs_out = None

        if self.facs_decoder is not None:
            facs_out, dec_h, _, _ = self.facs_decode(encoder_output=encoder_output, 
                                                     text_encoder_output=text_encoder_output,
                                                     src_mask=src_mask, text_src_mask=text_src_mask,
                                                     facs_trg_input=facs_trg_input,
                                                     facs_trg_mask=facs_trg_mask)
        

        return skel_out, nonreg_out, facs_out

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """

        encode_output = self.encoder(self.src_embed(src), src_length, src_mask)

        return encode_output


    def encode_text(self, text_src: Tensor, text_src_length: Tensor, text_src_mask: Tensor):
        """
        Encodes the source sentence.

        :param text_src:
        :param text_src_length:
        :param text_src_mask:
        :return: encoder outputs (output, hidden_concat)
        """

        encode_output = self.text_encoder(self.text_src_embed(text_src), text_src_length, text_src_mask)
    
        return encode_output


    def decode(self, encoder_output: Tensor, text_encoder_output:Tensor,
               src_mask: Tensor, trg_input: Tensor, text_src_mask:Tensor, 
               trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor):

        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        # Enbed the target using a linear layer
        trg_embed = self.trg_embed(trg_input)
        
        # Apply decoder to the embedded target
        decoder_output = self.decoder(trg_embed=trg_embed, encoder_output=encoder_output,
                               src_mask=src_mask, trg_mask=trg_mask,
                               text_encoder_output=text_encoder_output, text_src_mask=text_src_mask)

        return decoder_output

    def nonreg_decode(self, encoder_output: Tensor,
               src_mask: Tensor, nonreg_trg_input: Tensor,
               nonreg_trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor):

        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        # Enbed the target using a linear layer
        trg_embed = self.nonreg_trg_embed(nonreg_trg_input)
        
        # Apply decoder to the embedded target
        decoder_output = self.nonreg_decoder(trg_embed=trg_embed, encoder_output=encoder_output,
                               src_mask=src_mask,trg_mask=nonreg_trg_mask)

        return decoder_output
    
    def facs_decode(self, encoder_output: Tensor, text_encoder_output:Tensor,
               src_mask: Tensor, text_src_mask:Tensor, 
               facs_trg_input: Tensor, facs_trg_mask: Tensor = None) \
        -> (Tensor, Tensor, Tensor, Tensor):

        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """

        # Enbed the target using a linear layer
        trg_embed = self.facs_trg_embed(facs_trg_input)
        
        # Apply decoder to the embedded target
        decoder_output = self.facs_decoder(trg_embed=trg_embed, encoder_output=encoder_output,
                               src_mask=src_mask, trg_mask=facs_trg_mask,
                               text_encoder_output=text_encoder_output, text_src_mask=text_src_mask)
        
        # decoder_output = self.facs_decoder(trg_embed=trg_embed, encoder_output=encoder_output,
        #                        src_mask=src_mask,trg_mask=facs_trg_mask)

        return decoder_output

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module, nonreg_loss_function: nn.Module,
     xentloss_weight:int=0.0, regloss_weight:int=1.0, noise_rate:int = 0, 
     bceloss_weight:int=0.0, facs_loss_function: nn.Module = None) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # Forward through the batch input
        skel_out, nonreg_out, facs_out = self.forward(
            src=batch.src, trg_input=batch.trg_input, nonreg_trg_input=batch.nonreg_trg_inp,
            src_mask=batch.src_mask, src_lengths=batch.src_lengths,
            trg_mask=batch.trg_mask, nonreg_trg_mask=batch.nonreg_trg_mask, noise_rate=noise_rate,
            facs_trg_input=batch.facs_trg_inp, facs_trg_mask=batch.facs_trg_mask,
            text_src=batch.text_src, text_src_mask=batch.text_src_mask)

        # compute batch loss using skel_out and the batch target
        batch_loss = (regloss_weight)*loss_function(skel_out, batch.trg) + \
        xentloss_weight*nonreg_loss_function(nonreg_out, batch.nonreg_trg) + \
        bceloss_weight*facs_loss_function(facs_out, batch.facs_trg, batch.facs_trg_mask)

        # If gaussian noise, find the noise for the next epoch
        if self.gaussian_noise:
            # Calculate the difference between prediction and GT, to find STDs of error
            with torch.no_grad():
                noise = skel_out.detach() - batch.trg.detach()

            if self.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                noise = noise[:, :, :noise.shape[2] // (self.future_prediction)]

        else:
            noise = None

        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss, noise, skel_out

    def run_batch(self, batch: Batch, max_output_length: int,) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """
        # First encode the batch, as this can be done in all one go
        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        text_encoder_output, text_encoder_hidden = self.encode_text(text_src=batch.text_src, 
                                                                    text_src_length=None, 
                                                                    text_src_mask=batch.text_src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        # Then decode the batch separately, as needs to be done iteratively
        # greedy decoding
        stacked_output, stacked_attention_scores = greedy(
                encoder_output=encoder_output,
                text_encoder_output=text_encoder_output,
                src_mask=batch.src_mask,
                text_src_mask=batch.text_src_mask,
                embed=self.trg_embed,
                decoder=self.decoder,
                trg_input=batch.trg_input,
                model=self)

        return stacked_output, stacked_attention_scores

    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                   self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                nonreg_trg_vocab: Vocabulary = None,
                text_src_vocab: Vocabulary = None,
                ) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :return: built and initialized model
    """

    full_cfg = cfg
    cfg = cfg["model"]

    src_padding_idx = 0#src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = 0
    nonreg_trg_padding_idx = nonreg_trg_vocab.stoi[PAD_TOKEN]
    text_src_padding_idx = text_src_vocab.stoi[PAD_TOKEN] if text_src_vocab is not None else None
    
    # Input target size is the joint vector length plus one for counter
    in_trg_size = cfg["trg_size"] + 1
    
    # Output target size is the joint vector length plus one for counter
    out_trg_size = cfg["trg_size"] + 1
    
    nonreg_out_trg_size = len(nonreg_trg_vocab) ##### 4146, 5234
    nonreg_in_trg_size = len(nonreg_trg_vocab)

    text_src_size = len(text_src_vocab) if text_src_vocab is not None else None
    
    just_count_in = cfg.get("just_count_in", False)
    future_prediction = cfg.get("future_prediction", 0)

    #  Just count in limits the in target size to 1
    if just_count_in:
        in_trg_size = 1

    # Future Prediction increases the output target size
    if future_prediction != 0:
        # Times the trg_size (minus counter) by amount of predicted frames, and then add back counter
        out_trg_size = (out_trg_size - 1 ) * future_prediction + 1


    if cfg["encoder"]["type"] == 'transformer':
        # Transformer Encoder
        src_embed = nn.Linear(cfg["src_size"], cfg["encoder"]["embeddings"]["embedding_dim"])
        encoder = TransformerEncoder(**cfg["encoder"],
                                    emb_size=cfg["encoder"]["embeddings"]["embedding_dim"],
                                    emb_dropout=cfg["encoder"]["embeddings"]["dropout"])
        src_emb_size = cfg["encoder"]["embeddings"]["embedding_dim"]
    else:
        src_embed = nn.Identity()
        encoder = MockEncoder(**cfg["encoder"])
        src_emb_size = None
    # else:
    #     encoder = WhisperEncoder(**cfg["encoder"])

    if "text_encoder" in cfg and cfg["text_encoder"]["type"] == 'transformer':
        # Transformer Encoder
        text_src_embed = nn.Linear(cfg["text_src_size"], cfg["text_encoder"]["embeddings"]["embedding_dim"])
        text_encoder = TransformerEncoder(**cfg["text_encoder"],
                                    emb_size=cfg["text_encoder"]["embeddings"]["embedding_dim"],
                                    emb_dropout=cfg["text_encoder"]["embeddings"]["dropout"])
        text_emb_size = cfg["text_encoder"]["embeddings"]["embedding_dim"]
    elif "text_encoder" in cfg and cfg["text_encoder"]["type"] == 'RawTextEncoder':
        text_src_embed = Embeddings(
            **cfg["text_encoder"]["embeddings"], vocab_size=len(text_src_vocab),#4146,
            padding_idx=text_src_padding_idx)
        text_encoder = TransformerEncoder(**cfg["text_encoder"],
                                    emb_size=cfg["text_encoder"]["embeddings"]["embedding_dim"],
                                    emb_dropout=cfg["text_encoder"]["embeddings"]["dropout"])
        text_emb_size = cfg["text_encoder"]["embeddings"]["embedding_dim"]
    else:
        text_src_embed = nn.Identity()
        text_encoder = MockEncoder(**cfg["encoder"])
        text_emb_size = None

    if src_emb_size is not None and text_emb_size is not None:
        assert src_emb_size == text_emb_size, "src_emb_size must be equal to text_emb_size"
        emb_size = src_emb_size
    else:
        emb_size = src_emb_size if src_emb_size is not None else text_emb_size
 
    # TODO: Move this into try-catch for handling decoder as mock case
    dec_dropout = cfg["decoder"].get("dropout", 0.) # Dropout
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    decoder_trg_trg = cfg["decoder"].get("decoder_trg_trg", True)
    
    # Transformer Pose Decoder
    if "decoder" not in cfg or cfg["decoder"]["type"] == "transformer":
        trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_linear.out_features, emb_dropout=dec_emb_dropout,
            trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)
    elif cfg["decoder"]["type"] == "JointTransformerDecoder":
        trg_linear = nn.Linear(in_trg_size, cfg["decoder"]["embeddings"]["embedding_dim"])
        decoder = JointTransformerDecoder(
            **cfg["decoder"], encoder=encoder, text_encoder=text_encoder,
            vocab_size=len(trg_vocab), emb_size=trg_linear.out_features, 
            emb_dropout=dec_emb_dropout, trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)
    else:
        trg_linear = nn.Identity()
        trg_emb_size = emb_size
        decoder = MockDecoder(**cfg["decoder"], encoder=encoder, text_encoder=text_encoder,
            vocab_size=len(trg_vocab), emb_size=trg_emb_size, 
            emb_dropout=dec_emb_dropout, trg_size=out_trg_size, decoder_trg_trg_=decoder_trg_trg)

    
    # Transformer Text Decoder
    if "nonreg_decoder" not in cfg or cfg["nonreg_decoder"]["type"] == "transformer":
        nonreg_trg_embed = Embeddings(
            **cfg["nonreg_decoder"]["embeddings"], vocab_size=len(nonreg_trg_vocab),#4146,
            padding_idx=nonreg_trg_padding_idx)
        nonreg_decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(nonreg_trg_vocab),
            emb_size=nonreg_trg_embed.embedding_dim, emb_dropout=dec_emb_dropout,
            trg_size=len(nonreg_trg_vocab),
            decoder_trg_trg_=True)
    else:
        nonreg_trg_emb_size = emb_size
        # Cannot make this identity, because the embeddings for padding_idx are set to zero
        # in the model initialization code.
        nonreg_trg_embed = Embeddings(nonreg_trg_emb_size, vocab_size=len(nonreg_trg_vocab), 
                                      padding_idx=nonreg_trg_padding_idx)
        nonreg_decoder = MockDecoder(
            **cfg["nonreg_decoder"], encoder=encoder, vocab_size=len(nonreg_trg_vocab),
            emb_size=nonreg_trg_embed.embedding_dim, emb_dropout=dec_emb_dropout,
            trg_size=len(nonreg_trg_vocab),
            decoder_trg_trg_=True)
    
    if "facs_decoder" in cfg and cfg["facs_decoder"]["type"] == "transformer":
        facs_trg_size = cfg["facs_decoder"]["trg_size"]
        facs_embed = nn.Linear(facs_trg_size, cfg["facs_decoder"]["embeddings"]["embedding_dim"])
        facs_decoder = FACSDecoder(
            **cfg["facs_decoder"], encoder=encoder, vocab_size=facs_trg_size,
            emb_size=facs_embed.out_features, emb_dropout=dec_emb_dropout,
            decoder_trg_trg_=True)
    elif "facs_decoder" in cfg and cfg["facs_decoder"]["type"] == "JointTransformerDecoder":
        facs_trg_size = cfg["facs_decoder"]["trg_size"]
        facs_embed = nn.Linear(facs_trg_size, cfg["facs_decoder"]["embeddings"]["embedding_dim"])
        facs_decoder = JointTransformerDecoder(
            **cfg["facs_decoder"], encoder=encoder, vocab_size=facs_trg_size,
            emb_size=facs_embed.out_features, emb_dropout=dec_emb_dropout,
            decoder_trg_trg_=True)
    else:
        facs_embed = None
        facs_decoder = None

    
    # Cross Modal Discriminator 
    if cfg["encoder"]["type"] == "transformer":
        classifier = ClassifierLayers(**cfg["encoder"],
                                src_size=cfg["src_size"],
                                trg_size=cfg["trg_size"],
                                pose_time_dim=full_cfg["data"]["num_sec"]*full_cfg["data"]["trg_fps"],
                                aud_time_dim = full_cfg["data"]["num_sec"]*full_cfg["data"]["src_fps"],
                                emb_size=src_emb_size,
                                emb_dropout=cfg["encoder"]["embeddings"]["dropout"])
    else:
        classifier = MockClassifier(**cfg["encoder"],
                                src_size=cfg["src_size"],
                                trg_size=cfg["trg_size"],
                                pose_time_dim=full_cfg["data"]["num_sec"]*full_cfg["data"]["trg_fps"],
                                aud_time_dim = full_cfg["data"]["num_sec"]*full_cfg["data"]["src_fps"],
                                emb_size=None,
                                emb_dropout=None)
    
    # Define the model
    model = Model(encoder=encoder,
                  text_encoder=text_encoder,
                  decoder=decoder,
                  nonreg_decoder=nonreg_decoder,
                  src_embed=src_embed,
                  text_src_embed=text_src_embed,
                  trg_embed=trg_linear,
                  nonreg_trg_embed=nonreg_trg_embed,
                  src_vocab=src_vocab,
                  trg_vocab=trg_vocab,
                  nonreg_trg_vocab=nonreg_trg_vocab,
                  cfg=full_cfg,
                  in_trg_size=in_trg_size,
                  out_trg_size=out_trg_size,
                  facs_embed=facs_embed,
                  facs_decoder=facs_decoder)

    # Custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx, nonreg_trg_padding_idx)

    return model, classifier