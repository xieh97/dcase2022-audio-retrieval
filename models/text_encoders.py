import torch
import torch.nn as nn


class WordEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(WordEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.embedding = nn.Embedding(kwargs["num_word"], kwargs["embed_dim"])

        if kwargs.get("word_embeds", None) is not None:
            self.load_prior_embedding(kwargs["word_embeds"])
        else:
            nn.init.kaiming_uniform_(self.embedding.weight)

        for param in self.embedding.parameters():
            param.requires_grad = kwargs.get("trainable", False)

    def load_prior_embedding(self, weight):
        assert weight.shape[0] == self.embedding.weight.size()[0], "vocabulary size mismatch!"

        weight = torch.as_tensor(weight).float()
        self.embedding.weight = nn.Parameter(weight)

    def forward(self, text_vecs, text_lens):
        """
        :param text_vecs: tensor, (batch_size, text_max_len).
        :param text_lens: numpy 1D-array, (batch_size,).
        :return: tensor, (batch_size, text_max_len, embed_dim).
        """

        text_lens = torch.as_tensor(text_lens)
        batch_size, text_max_len = text_vecs.size()

        text_embeds = self.embedding(text_vecs)

        mask = torch.arange(text_max_len, device="cpu").repeat(batch_size).view(batch_size, text_max_len)
        mask = (mask < text_lens.view(-1, 1)).to(text_embeds.device)

        text_embeds = text_embeds * mask.unsqueeze(-1)

        return text_embeds


class TextEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(TextEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.word_enc = WordEncoder(**kwargs["word_enc"])

    def forward(self, text_vecs, text_lens):
        """
        :param text_vecs: tensor, (batch_size, text_max_len).
        :param text_lens: numpy 1D-array, (batch_size,).
        :return: tensor, (batch_size, embed_dim).
        """

        text_embeds = self.word_enc(text_vecs, text_lens)

        text_embeds = torch.mean(text_embeds, dim=1, keepdim=False)

        return text_embeds
