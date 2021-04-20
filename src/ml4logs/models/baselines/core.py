# ===== IMPORTS =====
# === Standard library ===
# === Thirdparty ===
import torch
import torch.utils.data as tdata
import torch.nn.utils.rnn as tutilsrnn

# === Local ===


# ===== CLASSES =====
class SequenceDataset(tdata.Dataset):
    def __init__(self, *args):
        self._data = tuple(zip(*args))

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


class SeqModel(torch.nn.Module):
    def __init__(self, f_dim, n_lstm_layers=1,
                 n_hidden_linears=2, linear_width=300, linear_norm=False,
                 linear_out_dim=1):
        super().__init__()
        self._lstm = torch.nn.LSTM(f_dim, f_dim, batch_first=True,
                                   num_layers=n_lstm_layers)
        linears = [torch.nn.Linear(f_dim, linear_width)]
        if linear_norm:
            linears.append(torch.nn.LayerNorm(linear_width))
        linears.append(torch.nn.LeakyReLU())
        for _ in range(n_hidden_linears):
            linears.append(torch.nn.Linear(linear_width, linear_width))
            if linear_norm:
                linears.append(torch.nn.LayerNorm(linear_width))
            linears.append(torch.nn.LeakyReLU())
        linears.append(torch.nn.Linear(linear_width, linear_out_dim))
        self._linears = torch.nn.Sequential(*linears)

    def forward(self, X):
        out, _ = self._lstm(X)
        out, lengths = tutilsrnn.pad_packed_sequence(X, batch_first=True)
        return self._linears(out)
