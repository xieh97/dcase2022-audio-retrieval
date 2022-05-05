import torch.nn as nn
import torch.nn.functional as F

from models import audio_encoders, text_encoders


class CRNNWordModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super(CRNNWordModel, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.audio_enc = audio_encoders.CRNNEncoder(**kwargs["audio_enc"])

        self.text_enc = text_encoders.TextEncoder(**kwargs["text_enc"])

    def forward(self, audio_vecs, text_vecs, text_lens):
        """
        :param audio_vecs: tensor, (batch_size, time_steps, Mel_bands).
        :param text_vecs: tensor, (batch_size, text_max_len).
        :param text_lens: numpy 1D-array, (batch_size,).
        """

        audio_embeds = self.audio_branch(audio_vecs)

        text_embeds = self.text_branch(text_vecs, text_lens)

        # audio_embeds: [N, E]    text_embeds: [N, E]
        return audio_embeds, text_embeds

    def audio_branch(self, audio_vecs):
        audio_embeds = self.audio_enc(audio_vecs)

        if self.kwargs.get("out_normd", None) == "L2":
            audio_embeds = F.normalize(audio_embeds, p=2.0, dim=-1)

        return audio_embeds

    def text_branch(self, text_vecs, text_lens):
        text_embeds = self.text_enc(text_vecs, text_lens)

        if self.kwargs.get("out_normd", None) == "L2":
            text_embeds = F.normalize(text_embeds, p=2.0, dim=-1)

        return text_embeds
