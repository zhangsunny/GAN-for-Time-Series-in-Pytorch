"""
使用LSTM构成的生成器和判别器
"""
from utils.manage_import import *


class LSTMGenerator(nn.Module):
    def __init__(self, input_shape, output_size,
                 hidden_size=64, num_layers=3):
        """
        Args:
            input_shape: (time_step, input_size)
            output_size: 生成数据的维度
            hidden_size: RNN Cell的size
            num_layers: LSTM层数
        """
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_step, self.input_size = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*self.time_step, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(in_features=256,
                      out_features=self.time_step*self.output_size),
            nn.BatchNorm1d(self.time_step*self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """输入数据为(batch_size, time_step, input_size)
        如果输入为二维数据，会被转换为三维，input_size=1
        """
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=2)
        batch_size = x.shape[0]
        h_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        c_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        pred = self.fc(output.contiguous().view(batch_size, -1))
        pred = pred.view(batch_size, self.time_step, self.output_size)
        return pred

    def init_variable(self, *args):
        return torch.randn(*args).to(self.device)


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_shape, hidden_size=64, num_layers=3):
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_step, self.input_size = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu_slope = 0.2
        self.drop_rate = 0.5
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*self.time_step, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(self.relu_slope),
            nn.Dropout(self.drop_rate),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(self.relu_slope),
            nn.Dropout(self.drop_rate),
            nn.Linear(in_features=256,
                      out_features=self.time_step),
            nn.BatchNorm1d(self.time_step),
            nn.LeakyReLU(self.relu_slope),
            nn.Dropout(self.drop_rate),
            nn.Linear(self.time_step, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=2)
        batch_size = x.shape[0]
        h_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        c_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        pred = self.fc(output.contiguous().view(batch_size, -1))
        return pred

    def init_variable(self, *args):
        return torch.randn(*args).to(self.device)


class LSTMGenerator2(nn.Module):
    def __init__(self, input_shape, output_size,
                 hidden_size=64, num_layers=3):
        """
        Args:
            input_shape: (time_step, input_size)
            output_size: 生成数据的维度
            hidden_size: RNN Cell的size
            num_layers: LSTM层数
        """
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_step, self.input_size = input_shape
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.relu_slope = 0.2
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*self.time_step, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(self.relu_slope),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(self.relu_slope),
            nn.Linear(in_features=256,
                      out_features=self.time_step*self.output_size),
            nn.BatchNorm1d(self.time_step*self.output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """输入数据为(batch_size, time_step, input_size)
        如果输入为二维数据，会被转换为三维，input_size=1
        """
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=2)
        batch_size = x.shape[0]
        h_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        c_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        pred = self.fc(output.contiguous().view(batch_size, -1))
        pred = pred.view(batch_size, self.time_step, self.output_size)
        return pred

    def init_variable(self, *args):
        return torch.randn(*args).to(self.device)


class LSTMCritic(nn.Module):
    def __init__(self, input_shape, hidden_size=64, num_layers=3):
        super().__init__()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.time_step, self.input_size = input_shape
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.relu_slope = 0.2
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size*self.time_step, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(self.relu_slope),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(self.relu_slope),
            nn.Linear(in_features=256,
                      out_features=self.time_step),
            nn.BatchNorm1d(self.time_step),
            nn.LeakyReLU(self.relu_slope),
            nn.Linear(self.time_step, 1),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = torch.unsqueeze(x, dim=2)
        batch_size = x.shape[0]
        h_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        c_0 = self.init_variable(
            self.num_layers, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        pred = self.fc(output.contiguous().view(batch_size, -1))
        return pred

    def init_variable(self, *args):
        return torch.randn(*args).to(self.device)
