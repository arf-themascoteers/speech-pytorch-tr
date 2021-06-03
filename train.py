import torch
import torch.nn.functional as F
import random
import simple_net
import speechdata


def train():
    working_set = speechdata.SubsetSC("training")
    working_set = speechdata.refine(working_set)
    random.shuffle(working_set)
    N_DATA = len(working_set)
    labels = speechdata.get_label_array(working_set)
    model = simple_net.SimpleNet()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    NUM_EPOCHS = 20
    BATCH_SIZE = 100
    BATCH_RUNS = N_DATA // BATCH_SIZE
    if N_DATA % BATCH_SIZE != 0:
        BATCH_RUNS = BATCH_RUNS + 1
    for epoch in range(NUM_EPOCHS):
        START_INDEX = 0
        for run in range(BATCH_RUNS):
            optimizer.zero_grad()
            START_INDEX = run * BATCH_SIZE
            END_INDEX = START_INDEX + BATCH_SIZE - 1
            if END_INDEX >= N_DATA - 1:
                END_INDEX = N_DATA - 1
            vectors = [working_set[i][0] for i in range(START_INDEX, END_INDEX+1)]
            vectors = torch.cat(vectors)
            label_true = [working_set[i][2] for i in range(START_INDEX, END_INDEX+1)]
            y_true = torch.tensor([labels.index(label) for label in label_true])
            y_pred = model(vectors)

            loss = F.nll_loss(y_pred, y_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    torch.save(model.state_dict(), 'models/linear.h5')

train()
exit(0)