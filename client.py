import hashlib
import os
import pickle
import torch
import dgl
import copy
from torch_geometric.utils import to_dense_adj
from torch_geometric.utils import add_self_loops
from models import Split_model
import torch.nn as nn
from torch.autograd import Variable
def hash_batch(batch):
    hash_obj = hashlib.sha256()
    for data in batch.to_data_list():
        hash_obj.update(data.edge_index.cpu().numpy().tobytes())
        if data.x is not None:
            hash_obj.update(data.x.cpu().numpy().tobytes())
    return hash_obj.hexdigest()

def collate_pyg_to_dgl(batch):
    dir_path = os.path.join(os.path.dirname(__file__), '..', 'preprocessed_batch')
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    file_name = os.path.join(dir_path, f"{hash_batch(batch)}.pkl")
    filtered_data_list = [data for data in batch.to_data_list() if data.edge_index.size(1) > 0]
    valid_indices = [i for i, data in enumerate(batch.to_data_list()) if data.edge_index.size(1) > 0]
    max_nodes = max([data.num_nodes for data in filtered_data_list], default=0)
    E = []
    U = []
    lengths = []
    for data in filtered_data_list:
        N = data.num_nodes
        edge_index, _ = add_self_loops(data.edge_index)
        adj = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0)
        D = torch.diag(torch.sum(adj, dim=1))
        L = D - adj
        e, u = torch.linalg.eigh(L)

        pad_e = e.new_zeros([max_nodes])
        pad_e[:N] = e

        pad_u = u.new_zeros([max_nodes, max_nodes])
        pad_u[:N, :N] = u

        E.append(pad_e)
        U.append(pad_u)
        lengths.append(N)

    E = torch.stack(E)
    U = torch.stack(U)
    lengths = torch.tensor(lengths)

    graphs = []
    for data in filtered_data_list:
        edge_index = data.edge_index.cpu()
        num_nodes = data.num_nodes if data.x is not None else (edge_index.max().item() + 1)
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)
        if data.x is not None:
            g.ndata['feat'] = data.x.cpu()
        graphs.append(g)

    g = dgl.batch(graphs)

    with open(file_name, 'wb') as f:
        pickle.dump((E, U, g, lengths), f)

    return E, U, g, lengths, valid_indices

class Client_GC():
    def __init__(self, model, client_id, client_name, train_size, dataLoader, optimizer, args):
        self.model = model.to(args.device)
        self.id = client_id
        self.name = client_name
        self.train_size = train_size
        self.dataLoader = dataLoader
        self.optimizer = optimizer
        self.args = args
        self.device = args.device

        self.W = {key: value for key, value in self.model.named_parameters()}
        self.dW = {key: torch.zeros_like(value) for key, value in self.model.named_parameters()}
        self.W_old = {key: value.data.clone() for key, value in self.model.named_parameters()}
        self.gconvNames = None

        self.train_stats = ([0], [0], [0], [0])
        self.weightsNorm = 0.
        self.gradsNorm = 0.
        self.convGradsNorm = 0.
        self.convWeightsNorm = 0.
        self.convDWsNorm = 0.

        self.train_preprocessed_batches = []
        self.test_preprocessed_batches = []
        self.val_preprocessed_batches = []
        self.pm_train = []
        self.lamda = 0
        self.train_samples = 0
        self.track = []
        self.tau = args.tau_weight
        self.momentum = args.momentum
        self.global_consensus = None

        self.current_mean = torch.zeros(args.hidden)
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)
        self.local_consensus = nn.Parameter(Variable(torch.zeros(args.hidden)))
        self.opt_local_consensus = torch.optim.SGD([self.local_consensus], lr=self.args.lr)

    def local_train(self, local_epoch):
        """ For self-train & FedAvg """
        if isinstance(self.model, Split_model):
            train_stats = train_gc_SSP(self, self.model, self.dataLoader, local_epoch, self.args.device, self.train_preprocessed_batches)

        self.train_stats = train_stats
        self.weightsNorm = torch.norm(flatten(self.W)).item()

    def evaluate(self):
        return eval_gc_test_SSP(self.model, self.args.device, self)

    def set_parameters_SSP(self, global_model):
        for (new_name, new_param), (old_name, old_param) in zip(global_model.named_parameters(), self.model.named_parameters()):
            if 'encoder' in new_name and 'atom' not in new_name:
                old_param.data = new_param.data.clone()

def flatten(w):
    return torch.cat([v.flatten() for v in w.values()])

def train_gc_SSP(client, model, dataloaders, local_epoch, device, train_preprocessed_batches):
    losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
    train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']

    for epoch in range(local_epoch):
        model.train()
        total_loss = 0.
        ngraphs = 0
        acc_sum = 0

        for batch in train_preprocessed_batches:
            e, u, g, length, label, _ = batch
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                          weight_decay=5e-4)
            optimizer.zero_grad()
            client.current_mean.zero_()
            client.num_batches_tracked.zero_()
            x = g.ndata['feat']
            rep, pred = model(e, u, g, length, x)
            current_mean = torch.mean(rep, dim=0).to(device)
            client.current_mean = client.current_mean.to(device)
            client.local_consensus = client.local_consensus.to(device)
            if client.num_batches_tracked is not None:
                client.num_batches_tracked.add_(1)
            client.current_mean = (1 - client.momentum) * client.current_mean + client.momentum * current_mean
            if client.global_consensus is not None:
                mse_loss = torch.mean(0.5 * (client.current_mean - client.global_consensus)**2)
                pred_pgpa = client.model.head(rep + client.local_consensus)
                loss = client.model.loss(pred_pgpa, label)
                loss = loss + mse_loss * client.tau
            else:
                pred_pgpa = client.model.head(rep)
                loss = client.model.loss(pred_pgpa, label)

            pred1 = torch.softmax(pred_pgpa, dim=1)
            pred_labels = torch.argmax(pred1, dim=1)
            correct_predictions = pred_labels.eq(label).sum().item()
            acc_sum += correct_predictions
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            client.opt_local_consensus.step()
            client.current_mean.detach_()
            total_loss += loss.item() * label.size(0)
            ngraphs += label.size(0)
        total_loss /= ngraphs
        acc = acc_sum / ngraphs
        loss_v, acc_v = eval_gc_val_SSP(model, device, client)
        loss_tt, acc_tt = eval_gc_test_SSP(model, device, client)
        losses_train.append(total_loss)
        accs_train.append(acc)
        losses_val.append(loss_v)
        accs_val.append(acc_v)
        losses_test.append(loss_tt)
        accs_test.append(acc_tt)

    return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val,
            'valAccs': accs_val,
            'testLosses': losses_test, 'testAccs': accs_test}



def eval_gc_test(model, device, client):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    test_preprocessed_batches = client.test_preprocessed_batches
    for batch in test_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x = g.ndata['feat']
        e, u, g, length, label = e.to(device), u.to(device), g.to(device), length.to(device), label.to(device)
        with torch.no_grad():
            rep, pred = client.model(e, u, g, length, x)
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs

def eval_gc_val(model, device, client):

    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    val_preprocessed_batches = client.val_preprocessed_batches
    for batch in val_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x= g.ndata['feat']
        e, u, g, length, label, x = e.to(device), u.to(device), g.to(device), length.to(device), label.to(
            device), x.to(device)
        with torch.no_grad():
            pred, rep, rep_base = client.model(e, u, g, length, x, is_rep=True, context=client.context)
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss / ngraphs, acc_sum / ngraphs

class clientAvgSSP(Client_GC):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.tau = args.tau_weight
        self.momentum = args.momentum
        self.global_consensus = None

        trainloader = self.load_train_data()
        for x, y in trainloader:
            if type(x) == type([]):
                x[0] = x[0].to(self.device)
            else:
                x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                rep = self.model.base(x).detach()
            break
        self.current_mean = torch.zeros_like(rep[0])
        self.num_batches_tracked = torch.tensor(0, dtype=torch.long, device=self.device)

        self.local_consensus = nn.Parameter(Variable(torch.zeros_like(rep[0])))
        self.opt_local_consensus = torch.optim.SGD([self.local_consensus], lr=self.learning_rate)

    def train_gc_SSP(client, model, dataloaders, local_epoch, device, train_preprocessed_batches):
        losses_train, accs_train, losses_val, accs_val, losses_test, accs_test = [], [], [], [], [], []
        train_loader, val_loader, test_loader = dataloaders['train'], dataloaders['val'], dataloaders['test']

        for epoch in range(local_epoch):
            model.train()
            total_loss = 0.
            ngraphs = 0
            acc_sum = 0

            for batch in train_preprocessed_batches:
                e, u, g, length, label, _ = batch
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8,
                                              weight_decay=5e-4)
                optimizer.zero_grad()
                client.current_mean.zero_()
                client.num_batches_tracked.zero_()
                x = g.ndata['feat']
                rep, pred = model(e, u, g, length, x)
                current_mean = torch.mean(rep, dim=0).to(device)
                client.current_mean = client.current_mean.to(device)
                client.local_consensus = client.local_consensus.to(device)
                if client.num_batches_tracked is not None:
                    client.num_batches_tracked.add_(1)
                client.current_mean = (1 - client.momentum) * client.current_mean + client.momentum * current_mean
                if client.global_consensus is not None:
                    mse_loss = torch.mean(0.5 * (client.current_mean - client.global_consensus) ** 2)
                    pred_pgpa = client.model.head(rep + client.local_consensus)
                    loss = client.model.loss(pred_pgpa, label)
                    loss = loss + mse_loss * client.tau
                else:
                    pred_pgpa = client.model.head(rep)
                    loss = client.model.loss(pred_pgpa, label)

                pred1 = torch.softmax(pred_pgpa, dim=1)
                pred_labels = torch.argmax(pred1, dim=1)
                correct_predictions = pred_labels.eq(label).sum().item()
                acc_sum += correct_predictions
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                client.opt_local_consensus.step()
                client.current_mean.detach_()
                total_loss += loss.item() * label.size(0)
                ngraphs += label.size(0)
            total_loss /= ngraphs
            acc = acc_sum / ngraphs
            loss_v, acc_v = eval_gc_val_SSP(model, device, client)
            loss_tt, acc_tt = eval_gc_test_SSP(model, device, client)
            losses_train.append(total_loss)
            accs_train.append(acc)
            losses_val.append(loss_v)
            accs_val.append(acc_v)
            losses_test.append(loss_tt)
            accs_test.append(acc_tt)

        return {'trainingLosses': losses_train, 'trainingAccs': accs_train, 'valLosses': losses_val,
                'valAccs': accs_val,
                'testLosses': losses_test, 'testAccs': accs_test}


def eval_gc_test_SSP(model, device, client):

    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    test_preprocessed_batches = client.test_preprocessed_batches
    for batch in test_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x = g.ndata['feat']
        e, u, g, length, label = e.to(device), u.to(device), g.to(device), length.to(device), label.to(device)
        with torch.no_grad():
            _, pred = client.model(e, u, g, length, x)
            acc_sum += pred.max(dim=1)[1].eq(label).sum().item()
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss/ngraphs, acc_sum/ngraphs


def eval_gc_val_SSP(model, device, client):
    model.eval()
    total_loss = 0.
    acc_sum = 0.
    ngraphs = 0
    val_preprocessed_batches = client.val_preprocessed_batches
    for batch in val_preprocessed_batches:
        e, u, g, length, label, num_graphs = batch
        x= g.ndata['feat']
        e, u, g, length, label, x = e.to(device), u.to(device), g.to(device), length.to(device), label.to(
            device), x.to(device)
        with torch.no_grad():
            rep, pred = client.model(e, u, g, length, x)
            pred1 = torch.softmax(pred, dim=1)
            pred_labels = torch.argmax(pred1, dim=1)
            correct_predictions = pred_labels.eq(label).sum().item()
            acc_sum += correct_predictions
            loss = model.loss(pred, label)
        total_loss += loss.item() * num_graphs
        ngraphs += num_graphs

    return total_loss / ngraphs, acc_sum / ngraphs