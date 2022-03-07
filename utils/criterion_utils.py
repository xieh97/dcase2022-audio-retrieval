import numpy as np
import torch
import torch.nn as nn


class TripletRankingLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()

        self.margin = margin

    def forward(self, audio_embeds, query_embeds, infos):
        """
        :param audio_embeds: tensor, (N, E).
        :param query_embeds: tensor, (N, E).
        :param infos: list of audio infos.
        :return:
        """
        N = audio_embeds.size(0)

        # Computes the triplet margin ranking loss for each anchor audio/query pair.
        # The impostor audio/query is randomly sampled from the mini-batch.
        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        for i in range(N):
            A_imp_idx = i
            while infos[A_imp_idx]["fid"] == infos[i]["fid"]:
                A_imp_idx = np.random.randint(0, N)

            Q_imp_idx = i
            while infos[Q_imp_idx]["fid"] == infos[i]["fid"]:
                Q_imp_idx = np.random.randint(0, N)

            anchor_score = score(audio_embeds[i], query_embeds[i])

            A_imp_score = score(audio_embeds[A_imp_idx], query_embeds[i])

            Q_imp_score = score(audio_embeds[i], query_embeds[Q_imp_idx])

            A2Q_diff = self.margin + Q_imp_score - anchor_score
            if (A2Q_diff.data > 0.).all():
                loss = loss + A2Q_diff

            Q2A_diff = self.margin + A_imp_score - anchor_score
            if (Q2A_diff.data > 0.).all():
                loss = loss + Q2A_diff

        loss = loss / N

        return loss


def score(audio_embed, query_embed):
    """
    Compute an audio-query score.

    :param audio_embed: tensor, (E, ).
    :param query_embed: tensor, (E, ).
    :return: similarity score: tensor, (1, ).
    """

    sim = torch.dot(audio_embed, query_embed)

    return sim
