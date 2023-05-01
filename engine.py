import torch
from torch import nn
import config
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup

# loss function is simple binary cross entropy loss
# need sigmoid to put probabilities in [0,1] interval


def loss_fn(outputs, targets):
    outputs = torch.squeeze(outputs)
    return nn.BCELoss()(nn.Sigmoid()(outputs), targets)


# computes perplexity on validation data
def calculate_perplexity(data_loader, model, device):
    model.eval()

    # tells Pytorch not to store values of intermediate computations for backward pass because we not gonna need gradients.
    with torch.no_grad():
        total_loss = 0
        for batch in data_loader:
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            total_loss += loss_fn(outputs, targets).item()

    model.train()

    return np.exp(total_loss / len(data_loader))


def train_loop(epochs, train_data_loader, val_data_loader, model, optimizer, device, scheduler=None):
    it = 1
    total_loss = 0
    curr_perplexity = None
    perplexity = None

    model.train()
    for epoch in range(epochs):
        print('Epoch: ', epoch + 1)
        for batch in train_data_loader:
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            token_type_ids = batch["token_type_ids"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.float)

            optimizer.zero_grad()

            # do forward pass, will save intermediate computations of the graph for later backprop use.
            outputs = model(ids, mask=mask, token_type_ids=token_type_ids)

            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

            # running backprop.
            loss.backward()

            # doing gradient descent step.
            optimizer.step()

            # we are logging current loss/perplexity in every 100 iteration
            if it % 100 == 0:

                # computing validation set perplexity in every 500 iteration.
                if it % 500 == 0:
                    curr_perplexity = calculate_perplexity(val_data_loader, model, device)

                    if scheduler is not None:
                        scheduler.step()

                    # making checkpoint of best model weights.
                    if not perplexity or curr_perplexity < perplexity:
                        torch.save(model.state_dict(), 'saved_model')
                        perplexity = curr_perplexity

                print('| It', it, '| Averageg Train Loss', total_loss / 100, '| Dev Perplexity', curr_perplexity)
                total_loss = 0

            it += 1


def run(model, device, train_data_loader, val_data_loader):
    EPOCHS = config.EPOCHS

    num_training_steps = int(len(train_data_loader) * EPOCHS)
    optimizer = AdamW(model.parameters(), lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    train_loop(config.EPOCHS, train_data_loader, val_data_loader, model, optimizer, device, scheduler)
