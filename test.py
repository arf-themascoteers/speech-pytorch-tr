import torch
import torch.nn.functional as F
import random
import simple_net
import speechdata


def test():
    working_set = speechdata.SubsetSC("validation")
    working_set = speechdata.refine(working_set)
    random.shuffle(working_set)
    labels = speechdata.get_label_array(working_set)
    model = simple_net.SimpleNet()
    model.load_state_dict(torch.load("models/linear.h5"))
    model.eval()
    correct = 0
    with torch.no_grad():
        for data in working_set:
            vectors = [data[0]]
            vectors = torch.cat(vectors)
            label_true = [data[2]]
            y_true = torch.tensor([labels.index(label) for label in label_true])
            y_pred = model(vectors)
            pred = y_pred.data.max(1, keepdim=True)[1]
            correct += pred.eq(y_true.data.view_as(pred)).sum()
    print(f"{correct} correct among {len(working_set)}")
test()
exit(0)