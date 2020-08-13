import tqdm
import dgl
import torch


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))

    wt_graphs, mut_graphs = map(list, zip(*graphs))

    wt_graphs, mut_graphs = dgl.batch(wt_graphs), dgl.batch(mut_graphs)

    wt_graphs.set_n_initializer(dgl.init.zero_initializer)
    wt_graphs.set_e_initializer(dgl.init.zero_initializer)

    mut_graphs.set_n_initializer(dgl.init.zero_initializer)
    mut_graphs.set_e_initializer(dgl.init.zero_initializer)

    batched_graph = [wt_graphs, mut_graphs]
    """
    print(batched_graph)
    print("---")

    print(wt_graphs)
    print(labels)
    print(torch.FloatTensor(labels))
    """
    return batched_graph, torch.FloatTensor(labels)
