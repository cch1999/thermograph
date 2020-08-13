import tqdm
<<<<<<< HEAD
import dgl
import torch

=======

class CheckPDBs(object):

    def __init__(self, df):

        self.pdb_path = "files/pdb/raw/"
        self.wt_path = "files/pdb/wt"
        self.mut_path = "files/pdb/mut"

        for index, row in df.iterrows():

            self.generate_wt(variant)
            self.generate_mut(variant)

    def generate_wt(self, variant):

        #Check raw pdb downloaded

        #Repair pdb

    def generate_mut(self, variant):

        #Run rosetta
>>>>>>> a43333aa926a87757ba0f52b6928565b4a3d4606

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

<<<<<<< HEAD
    batched_graph = [wt_graphs, mut_graphs]
    """
    print(batched_graph)
    print("---")

    print(wt_graphs)
    print(labels)
    print(torch.FloatTensor(labels))
    """
    return batched_graph, torch.FloatTensor(labels)
=======
    batched_graph = zip(wt_graphs, mut_graphs)

    return batched_graph, torch.stack(labels)
>>>>>>> a43333aa926a87757ba0f52b6928565b4a3d4606
