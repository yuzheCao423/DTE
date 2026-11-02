import torch
import torch.nn as nn
import lightning.pytorch as pl
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.distributions import StudentTOutput
from gluonts.model.forecast_generator import DistributionForecastGenerator
from typing import List, Callable
from models.Seq_TCN import Seq_TCN

def mean_abs_scaling(context, min_scale=1e-5):
    return context.abs().mean(1).clamp(min_scale, None).unsqueeze(1)

class NetworkWithDTE(nn.Module):
    def __init__(
        self,
        prediction_length: int,
        context_length: int,
        hidden_dimensions: List[int],
        distr_output=StudentTOutput(),
        batch_norm: bool = False,
        scaling: Callable = mean_abs_scaling,
        config: dict = None,  # 用于传递 Seq_TCN 的参数
    ) -> None:
        super().__init__()

        assert prediction_length > 0
        assert context_length > 0
        assert len(hidden_dimensions) > 0

        self.prediction_length = prediction_length
        self.context_length = context_length
        self.hidden_dimensions = hidden_dimensions
        self.distr_output = distr_output
        self.batch_norm = batch_norm
        self.scaling = scaling

        dimensions = [context_length] + hidden_dimensions[:-1]

        modules = []
        for in_size, out_size in zip(dimensions[:-1], dimensions[1:]):
            modules += [self.__make_lin(in_size, out_size), nn.ReLU()]
            if batch_norm:
                modules.append(nn.BatchNorm1d(out_size))
        modules.append(
            self.__make_lin(dimensions[-1], prediction_length * hidden_dimensions[-1])
        )

        self.model = Seq_TCN(
            input_size=1,
            output_size=4,
            hidden_size=hidden_dimensions[-1],
            context_length=context_length,
            prediction_length=prediction_length,
            cell_type=config['cell_type'],
            tangle_channels=config['tangle_channels'],
            tcn_channels=config['tcn_channels'],
            tcn_layers=config['tcn_layers'],
            rnn_hidden_size=config['rnn_hidden_size'],
            rnn_layers=config['rnn_layers'],
            embed_size=config['embed_size'],
            lee_size=config['lee_size'],
            learned_embedding_dim=config['learned_embedding_dim'],
        )

        self.args_proj = self.distr_output.get_args_proj(hidden_dimensions[-1])

    @staticmethod
    def __make_lin(dim_in, dim_out):
        lin = nn.Linear(dim_in, dim_out)
        torch.nn.init.uniform_(lin.weight, -0.07, 0.07)
        torch.nn.init.zeros_(lin.bias)
        return lin

    def forward(self, past_target):
        scale = self.scaling(past_target)
        scaled_past_target = past_target / scale
        nn_out = self.model(scaled_past_target)   # [batch, pred_len, hidden]
        nn_out_reshaped = nn_out.reshape(
            -1, self.prediction_length, self.hidden_dimensions[-1]
        )
        distr_args = self.args_proj(nn_out_reshaped)
        return distr_args, torch.zeros_like(scale), scale

    def get_predictor(self, input_transform, batch_size=32):
        return PyTorchPredictor(
            prediction_length=self.prediction_length,
            input_names=["past_target"],
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            forecast_generator=DistributionForecastGenerator(self.distr_output),
        )

class LightningNetworkWithDTE(NetworkWithDTE, pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        context = batch["past_target"]
        target = batch["future_target"]

        assert context.shape[-1] == self.context_length
        assert target.shape[-1] == self.prediction_length

        distr_args, loc, scale = self(context)
        distr = self.distr_output.distribution(distr_args, loc, scale)
        loss = -distr.log_prob(target)

        self.log('train_loss', loss.mean(), on_step=True, on_epoch=True,
                 prog_bar=True, logger=True, batch_size=batch["past_target"].shape[0])
        return loss.mean()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=7e-4)  # lr 可从 config 传入
        return optimizer