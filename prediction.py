
import numpy as np
import math

import torch
try:
    from torchtext.data import Dataset
except:
    from torchtext.legacy.data import Dataset

from helpers import bpe_postprocess, load_config, get_latest_checkpoint, \
    load_checkpoint, calculate_dtw, calculate_pck, fastest_dtw, calculate_dtw_parallel
from plot_videos_85 import alter_DTW_timing
from model_old import build_model, Model
from batch import Batch
from data_files import load_data, make_data_iter
from constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence

def get_mean_cov(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x.mean(axis = 0), np.cov(x, rowvar = False)


def cal_mean_for_length(x, device):
    mask = torch.tensor([len(x) for _x in x], device = device)
    x = pad_sequence(x, batch_first = True, padding_value = 0.)
    # x = x.sum(1) / mask.unsqueeze(1)
    x = rearrange(x, 'b t d -> (b t) d')
    
    return x

def eval_fgd(self, generated, reference, device = 'cpu'):
        assert type(generated) == list, 'Inputs must be list of tensors.'

        pred_z = self.encode(generated, device = device)
        real_z = self.encode(reference, device = device)

        pred_z, real_z = map(lambda x: cal_mean_for_length(x, device), [pred_z, real_z])
        # pred_z, real_z = map(lambda x: torch.vstack(x), [pred_z, real_z])
                
        (pred_mu, pred_sigma), (real_mu, real_sigma) = \
            map(lambda x: get_mean_cov(x), [pred_z, real_z])
        
        score = calculate_frechet_distance(pred_mu, pred_sigma, real_mu, real_sigma)
        
        return score

# Validate epoch given a dataset
def validate_on_data(model: Model,
                     data: Dataset,
                     batch_size: int,
                     max_output_length: int,
                     eval_metric: str,
                     loss_function: torch.nn.Module = None,
                     nonreg_loss_function: torch.nn.Module = None,
                     facs_loss_function: torch.nn.Module = None,
                     batch_type: str = "sentence",
                     type = "val",
                     BT_model = None,
                     max_feature_index=None):

    valid_iter = make_data_iter(
        dataset=data, batch_size=batch_size, batch_type=batch_type,
        shuffle=True, train=False)

    pad_index = 0
    nonreg_pad_index = model.nonreg_pad_index
    
    # disable dropout
    model.eval()

    # don't track gradients during validation
    with torch.no_grad():
        seen_files = {}

        valid_hypotheses = []
        valid_references = []
        valid_inputs = []
        file_paths = []
        all_dtw_scores = []
        all_pcks = []
    


        valid_loss = 0
        total_ntokens = 0
        total_nseqs = 0

        batches = 0
        for valid_batch in iter(valid_iter):
            # Extract batch
            batch = Batch(torch_batch=valid_batch,
                          pad_index = pad_index,
                          nonreg_pad_index = nonreg_pad_index,
                          model = model)
            targets = batch.trg
            
            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:
                # Get the loss for this batch
                batch_loss, _, _ = model.get_loss_for_batch(
                    batch, loss_function=loss_function, nonreg_loss_function=nonreg_loss_function,
                    facs_loss_function=facs_loss_function)
                
                # batch_loss, _, _ = model.get_loss_for_batch(
                #     batch=batch, loss_function=loss_function, nonreg_loss_function=nonreg_loss_function)

                valid_loss += batch_loss
                total_ntokens += batch.ntokens
                total_nseqs += batch.nseqs

            # If not just count in, run inference to produce translation videos
            if not model.just_count_in:
                # Run batch through the model in an auto-regressive format
                output, attention_scores = model.run_batch(
                                            batch=batch,
                                            max_output_length=max_output_length)
            
            # print('targets output')
            # print(targets.shape, output.shape)

            # If future prediction
            if model.future_prediction != 0:
                # Cut to only the first frame prediction + add the counter
                targets = torch.cat((targets[:, :, :targets.shape[2] // (model.future_prediction)], targets[:, :, -1:]),dim=2)

            # For just counter, the inference is the same as GTing
            if model.just_count_in:
                output = train_output

            # Add references, hypotheses and file paths to list
            valid_references.extend(targets)
            
            valid_hypotheses.extend(output)
            file_paths.extend(batch.file_paths)
            
            # Add the source sentences to list, by using the model source vocab and batch indices
            valid_inputs.extend(batch.src)
            
            # Calculate the full Dynamic Time Warping score - for evaluation
            
            dtw_score = calculate_dtw(targets.cpu(), output.cpu(), max_feature_index)
            # dtw_score = fastest_dtw(targets, output)
            all_dtw_scores.extend(dtw_score)
            
            # TODO: Commented this out since alter_DTW_timing is slow. Fix this later.
            # # Calculate the PCK for entire vid sequence
            for b in range(output.shape[0]):
                hyp, ref, _ = alter_DTW_timing(output[b], targets[b])
                pck = calculate_pck(hyp[:,:-1], ref[:,:-1])
                all_pcks.append(np.mean(pck))
            # all_pcks.extend([0]*output.shape[0])

            # Can set to only run a few batches
            # if batches == math.ceil(100/batch_size):
            #     break
            batches += 1
            
        # # Calculate the full Dynamic Time Warping score - for evaluation
        # dtw_score = fastest_dtw(torch.stack(valid_references), torch.stack(valid_hypotheses))
        # all_dtw_scores= dtw_score

        current_valid_score = np.mean(all_dtw_scores)
        
        
        # gd_scores = 

    return current_valid_score, valid_loss, valid_references, valid_hypotheses, \
           valid_inputs, all_dtw_scores, file_paths, all_pcks

