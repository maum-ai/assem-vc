# MIT License
#
# Copyright (c) 2020 Tomáš Nekvinda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch

class GuidedAttentionLoss(torch.nn.Module):
    """Wrapper around all loss functions including the loss of Tacotron 2.

    Details:
        - L2 of the prediction before and after the postnet.
        - Cross entropy of the stop tokens
        - Guided attention loss:
            prompt the attention matrix to be nearly diagonal, this is how people usualy read text
            introduced by 'Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention'
    Arguments:
        guided_att_steps -- number of training steps for which the guided attention is enabled
        guided_att_variance -- initial allowed variance of the guided attention (strictness of diagonal) 
        guided_att_gamma -- multiplier which is applied to guided_att_variance at every update_states call
    """

    def __init__(self, guided_att_steps, guided_att_variance, guided_att_gamma):
        super(GuidedAttentionLoss, self).__init__()
        self._g = guided_att_variance
        self._gamma = guided_att_gamma
        self._g_steps = guided_att_steps

    def forward(self, alignments, input_lengths, target_lengths, global_step):
        if self._g_steps < global_step: return 0
        self._g = self._gamma ** global_step
        # compute guided attention weights (diagonal matrix with zeros on a 'blurry' diagonal)
        weights = torch.zeros_like(alignments)
        for i, (f, l) in enumerate(zip(target_lengths, input_lengths)):
            grid_f, grid_l = torch.meshgrid(torch.arange(f, dtype=torch.float, device=f.device),
                                            torch.arange(l, dtype=torch.float, device=l.device))
            weights[i, :f, :l] = 1 - torch.exp(-(grid_l / l - grid_f / f) ** 2 / (2 * self._g ** 2))

            # apply weights and compute mean loss 
        loss = torch.sum(weights * alignments, dim=(1, 2))
        loss = torch.mean(loss / target_lengths.float())

        return loss