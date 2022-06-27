from util import load_data_snli
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.vocab import Vectors
from model import ESIM
from tqdm import tqdm

torch.manual_seed(42)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 32
HIDDEN_SIZE = 600  # every LSTM's(forward and backward) hidden size is half of HIDDEN_SIZE
EPOCHS = 20
DROPOUT_RATE = 0.5
LAYER_NUM = 1
LEARNING_RATE = 4e-4
PATIENCE = 5
CLIP = 10
EMBEDDING_SIZE = 300
# vectors = None

vectors = Vectors('glove.840B.300d.txt', "./data/")
freeze = False
data_path = "./data/snli_1.0"


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def eval(model, loss_func, data_iter, name, epoch=None, use_cache=False):
    if use_cache:
        model.load_state_dict(torch.load("best_model.ckpt"))
    model.eval()
    correct_num = 0
    err_num = 0
    total_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_iter):
            # premise, premise_lens = batch.premise
            # hypothesis, hypothesis_lens = batch.hypothesis
            # labels = batch.label
            (premise, hypothesis), (premise_len, hypothesis_len), labels = batch
            
            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            premise_len = premise_len.to(device)
            hypothesis_len = hypothesis_len.to(device)
            labels = labels.to(device)

            # output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            output = model(premise, premise_len, hypothesis, hypothesis_len)
            predicts = output.argmax(-1).reshape(-1)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            correct_num += (predicts == labels).sum().item()
            err_num += (predicts != labels).sum().item()
    total_loss = total_loss / len(data_iter)
    acc = correct_num / (correct_num + err_num)
    if epoch is not None:
        tqdm.write("Epoch: %d, %s Acc: %.3f, Loss %.3f" % (epoch + 1, name, acc, total_loss))
    else:
        tqdm.write("%s Acc: %.3f, Loss %.3f" % (name, acc, total_loss))
    return acc


def train(model, loss_func, train_iter, dev_iter, optimizer, epochs, patience=5, clip=5):
    best_acc = -1
    patience_counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_iter):
            # premise, premise_lens = batch.premise
            # hypothesis, hypothesis_lens = batch.hypothesis
            # labels = batch.label
            (premise, hypothesis), (premise_len, hypothesis_len), labels = batch

            premise = premise.to(device)
            hypothesis = hypothesis.to(device)
            premise_len = premise_len.to(device)
            hypothesis_len = hypothesis_len.to(device)
            labels = labels.to(device)

            model.zero_grad()
            # output = model(premise, premise_lens, hypothesis, hypothesis_lens)
            output = model(premise, premise_len, hypothesis, hypothesis_len)
            loss = loss_func(output, labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total_loss = total_loss / len(train_iter)
        tqdm.write("Epoch: %d, Train Loss: %.3f" % (epoch + 1, total_loss))

        acc = eval(model, loss_func, dev_iter, "Dev", epoch)
        if acc < best_acc:
            patience_counter += 1
        else:
            bast_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.ckpt")
        if patience_counter >= patience:
            tqdm.write("Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    train_iter, dev_iter, test_iter, vocab = load_data_snli(BATCH_SIZE, data_path=data_path)

    model = ESIM(vocab_size=len(vocab), num_labels=3, embed_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE,
                 dropout_rate=DROPOUT_RATE, layer_num=LAYER_NUM, pretrained_embed=None, freeze=freeze)
    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()

    model.to(device)

    train(model, loss_func, train_iter, dev_iter, optimizer, EPOCHS, PATIENCE, CLIP)
    eval(model, loss_func, test_iter, "Test", use_cache=True)