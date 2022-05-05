import numpy as np
import torch
import torch.nn as nn


class TripletRankingLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletRankingLoss, self).__init__()

        self.margin = margin

    def forward(self, audio_embeds, text_embeds, add_infos):
        """
        :param audio_embeds: tensor, (N, E).
        :param text_embeds: tensor, (N, E).
        :param add_infos: list of audio-text infos.
        :return:
        """
        N = audio_embeds.size(0)

        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        # Computes the triplet margin ranking loss for each anchor audio/text pair.
        # The impostor audio/text is randomly sampled from the mini-batch.
        for i in range(N):

            a = i  # index of imposter audio
            while add_infos[a]["fid"] == add_infos[i]["fid"]:
                a = np.random.randint(0, N)

            t = i  # index of imposter text
            while add_infos[t]["fid"] == add_infos[i]["fid"] or a == t:
                t = np.random.randint(0, N)

            S_ii = score(audio_embeds[i], text_embeds[i])
            S_ai = score(audio_embeds[a], text_embeds[i])
            S_it = score(audio_embeds[i], text_embeds[t])

            L_ai = S_ai - S_ii + self.margin
            if (L_ai.data > 0.).all():
                loss = loss + L_ai

            L_it = S_it - S_ii + self.margin
            if (L_it.data > 0.).all():
                loss = loss + L_it

        loss = loss / N

        return loss


def score(audio_embed, text_embed):
    """
    Compute an audio-text score.

    :param audio_embed: tensor, (E, ).
    :param text_embed: tensor, (E, ).
    :return: similarity score: tensor, (1, ).
    """

    sim = torch.dot(audio_embed, text_embed)

    return sim
