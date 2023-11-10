import torch

_STOCK_LOSSES = {
    "L1Loss": torch.nn.L1Loss,
    "MSELoss": torch.nn.MSELoss,
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "CTCLoss": torch.nn.CTCLoss,
    "NLLLoss": torch.nn.NLLLoss,
    "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
    "KLDivLoss": torch.nn.KLDivLoss,
    "BCELoss": torch.nn.BCELoss,
    "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
    "MarginRankingLoss": torch.nn.MarginRankingLoss,
    "HingeEmbeddingLoss": torch.nn.HingeEmbeddingLoss,
    "MultiLabelMarginLoss": torch.nn.MultiLabelMarginLoss,
    "HuberLoss": torch.nn.HuberLoss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss,
    "SoftMarginLoss": torch.nn.SoftMarginLoss,
    "MultiLabelSoftMarginLoss": torch.nn.MultiLabelSoftMarginLoss,
    "CosineEmbeddingLoss": torch.nn.CosineEmbeddingLoss,
    "MultiMarginLoss": torch.nn.MultiMarginLoss,
    "TripletMarginLoss": torch.nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": torch.nn.TripletMarginWithDistanceLoss,
}
