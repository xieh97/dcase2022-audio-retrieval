import torch
import torch.nn as nn


class WordEmbedding(nn.Module):

    def __init__(self, *args, **kwargs):
        super(WordEmbedding, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.embedding = nn.Embedding(kwargs["num_word"], kwargs["embed_dim"])

        if kwargs.get("word_embeds", None) is not None:
            self.load_pretrained_embedding(kwargs["word_embeds"])
        else:
            nn.init.kaiming_uniform_(self.embedding.weight)

        for para in self.embedding.parameters():
            para.requires_grad = kwargs.get("trainable", False)

    def load_pretrained_embedding(self, weight):
        assert weight.shape[0] == self.embedding.weight.size()[0], "vocabulary size mismatch!"

        weight = torch.as_tensor(weight).float()
        self.embedding.weight = nn.Parameter(weight)

    def forward(self, queries, query_lens):
        """
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, query_max_len, embed_dim).
        """

        query_lens = torch.as_tensor(query_lens)
        batch_size, query_max = queries.size()

        query_embeds = self.embedding(queries)

        mask = torch.arange(query_max, device="cpu").repeat(batch_size).view(batch_size, query_max)
        mask = (mask < query_lens.view(-1, 1)).to(query_embeds.device)

        query_embeds = query_embeds * mask.unsqueeze(-1)

        return query_embeds


class WordEncoder(nn.Module):

    def __init__(self, *args, **kwargs):
        super(WordEncoder, self).__init__()

        self.args = args
        self.kwargs = kwargs

        self.word_embedding = WordEmbedding(*args, **kwargs["word_embedding"])

    def forward(self, queries, query_lens):
        """
        :param queries: tensor, (batch_size, query_max_len).
        :param query_lens: list, [N_{1}, ..., N_{batch_size}].
        :return: (batch_size, embed_dim).
        """

        query_embeds = self.word_embedding(queries, query_lens)

        query_embeds = torch.mean(query_embeds, dim=1, keepdim=False)

        return query_embeds


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
