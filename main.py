from nn.nn import Predictor

#----Initialise protein graphs----

pg = ProteinGraph(granularity='CA', insertions=False, keep_hets=True,
          node_featuriser='meiler',
          get_contacts_path='~/projects/repos/getcontacts',
          pdb_dir='files/pdbs/',
          contacts_dir='files/contacts/',
          exclude_waters=True,
          covalent_bonds=True,
          include_ss=True,
          include_ligand=False,
          remove_string_labels=True)

#----Construst graphs----

train_graphs = []
test_graphs = []

for pdb_id in train_pdbs:

    wt_path = "files/wt/" + pdb_id + ".pdb"
    mut_path = "files/mut/" + pdb_id + ".pdb"

    train_graphs.append([pg.dgl_graph_from_pdb_code(file_path=wt_path), pg.dgl_graph_from_pdb_code(file_path=mut_path)])

for pdb_id in test_pdbs:

    wt_path = "files/wt/" + pdb_id + ".pdb"
    mut_path = "files/mut/" + pdb_id + ".pdb"

    test_graphs.append([pg.dgl_graph_from_pdb_code(file_path=wt_path), pg.dgl_graph_from_pdb_code(file_path=mut_path)])

#----Create dataloaders----

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

    batched_graph = zip(wt_graphs, mut_graphs)

    return batched_graph, torch.stack(labels)

train_data = list(zip(train_graphs, y_train))
test_data = list(zip(test_graphs, y_test))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True,
                         collate_fn=collate)

test_loader = DataLoader(test_data, batch_size=32, shuffle=True,
                         collate_fn=collate)

#----Inititialise model----

model = Predictor(in_feats, hidden_feats=[32, 32], batchnorm=[True, True], dropout=[0, 0], predictor_hidden_feats=32, n_tasks=6)