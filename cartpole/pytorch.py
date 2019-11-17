import torch
from torch import nn
from matplotlib import pyplot as plt

class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_1=nn.Linear(1, 10)
        self.linear_2=nn.Linear(10, 1)
        self.tanh = nn.Tanh()
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.1)

    def forward(self, input):
        hidden = self.linear_1(input)
        hidden = self.tanh(hidden)
        output = self.linear_2(hidden)
        return output

    def fit(self,x,y):
        loss = torch.mean((self.forward(x) - y) ** 2)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def fun(self):
        for p in self.parameters():
            print(p)

agent = Agent()


# need to init x
x = torch.tensor([0.01 * i for i in range(300)])
noise = torch.tensor([torch.normal(torch.tensor(0.1), torch.tensor(0.1))])
y = torch.sin(x) + noise
x = x.reshape(300, 1)
y = x.reshape(300,1)
for _ in range(100):
    agent.fit(x,y)


plt.plot(x.numpy(), y.numpy())
plt.plot(x.numpy(), agent(x).detach().numpy())
plt.show()