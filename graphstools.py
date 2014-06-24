import numpy as np
import cosmology
from sklearn.neighbors import BallTree as BallTree
list_edges = []
all_nodes_sorted = None
xyz = None

# X :  id, x1, x2, z, w, ids
col_id = 0
col_ra = 1
col_de = 2
col_z = 3
col_w = 4
col_ids_sorted = 5

nodes_used = []

min_dist_deg = 0.5

def get_nearest_nodes(node_id):


    xyz_this = xyz[nodes_used,:]
    all_nodes_sorted_this = all_nodes_sorted[nodes_used,:]
    BT = BallTree(xyz_this, leaf_size=5)
    n_connections=min([len(nodes_used),50])
    bt_dx,bt_id = BT.query(xyz[node_id,:],k=n_connections)
    
    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(  all_nodes_sorted[node_id,col_ra] , all_nodes_sorted[node_id,col_de]  )
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(  all_nodes_sorted_this[bt_id[0],col_ra] , all_nodes_sorted_this[bt_id[0],col_de]  )
    dxy = cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad)[0] * 180. / np.pi
    ids = all_nodes_sorted_this[bt_id[0]][:,col_ids_sorted]
      
    return ids,dxy

def check_criteria(node,candidate):

    pass


def connect_node(node_id):

    if len(nodes_used) == 0:
        nodes_used.append(node_id)
        print 'added node %d' % (node_id)
    else:
        ids, dxy = get_nearest_nodes(node_id)

        add_node=True
        for i in range(len(ids)):
            if (ids[i] in nodes_used) & (dxy[i] < min_dist_deg):
                print 'skipping node %d as it is close to %d with dx=%2.2f deg' % (node_id, ids[i], dxy[i])
                add_node=False
        if add_node:
            nodes_used.append(node_id)
            print 'added node %d min(dx)=%2.4f, len(nodes_used)=%d' % (node_id,min(dxy),len(nodes_used))


def get_graph(all_nodes):
    """
    X :  id, x1, x2, z, w, id_sorted
    """
    n_nodes = all_nodes.shape[0]
    sorting = np.argsort(all_nodes[:,col_w])

    global all_nodes_sorted
    all_nodes_sorted=all_nodes[sorting,:]
    all_nodes_sorted[:,col_ids_sorted]=np.arange(0,n_nodes)

    x,y,z = cosmology.spherical_to_cartesian_with_redshift(all_nodes_sorted[:,col_ra],all_nodes_sorted[:,col_de],all_nodes_sorted[:,col_z])
    global xyz
    xyz=np.concatenate([x[:,None],y[:,None],z[:,None]],axis=1)
    
    for node_id in range(n_nodes):

        connect_node(node_id)

    print 'using %d / %d'  % (len(nodes_used),n_nodes)

    import pylab as pl
    pl.scatter(all_nodes_sorted[:,col_ra],all_nodes_sorted[:,col_de],c='b')
    pl.scatter(all_nodes_sorted[nodes_used,col_ra],all_nodes_sorted[nodes_used,col_de],c='r')
    pl.show()


    import pdb; pdb.set_trace()

