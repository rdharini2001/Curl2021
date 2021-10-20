## Specify loss and optimization functions
criterion = nn.CrossEntropyLoss()
# specify optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
