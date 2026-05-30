from torch import nn


class LSTMForecaster(nn.Module):
    """
    LSTM-модель для временных рядов.
    Архитектура: LSTM -> последнее скрытое состояние -> Linear -> прогноз.
    Возвращает тензор формы (batch_size, 1) для совместимости с MSE Loss и API.
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        out, _ = self.rnn(x)  # out: (batch_size, seq_len, hidden_size)
        last_hidden = out[:, -1, :]  # (batch_size, hidden_size)
        pred = self.head(last_hidden)  # (batch_size, 1) ← без squeeze!
        return pred


class GRUForecaster(nn.Module):
    """
    GRU-модель для временных рядов.
    Архитектура аналогична LSTM, но с меньшим количеством параметров.
    Возвращает тензор формы (batch_size, 1).
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,  #  Выровнено с LSTM для честного сравнения
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]
        pred = self.head(last_hidden)
        return pred
