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

def get_nearest_nodes(node_id):


    xyz_this = xyz[nodes_used,:]
    all_nodes_sorted_this = all_nodes_sorted[nodes_used,:]
    BT = BallTree(xyz_this, leaf_size=5)
    n_connections=min([len(nodes_used),50])
    bt_dx,bt_id = BT.query(xyz[node_id,:],k=n_connections)
    
    halo1_ra_rad , halo1_de_rad = cosmology.deg_to_rad(  all_nodes_sorted[node_id,col_ra] , all_nodes_sorted[node_id,col_de]  )
    halo2_ra_rad , halo2_de_rad = cosmology.deg_to_rad(  all_nodes_sorted_this[bt_id[0],col_ra] , all_nodes_sorted_this[bt_id[0],col_de]  )
    # dxy = cosmology.get_angular_separation(halo1_ra_rad , halo1_de_rad , halo2_ra_rad , halo2_de_rad) * 180. / np.pi
    dxy = np.sqrt((halo1_ra_rad*180/np.pi - halo2_ra_rad*180/np.pi)**2 + (halo1_de_rad*180/np.pi - halo2_de_rad*180/np.pi)**2)
    ids = all_nodes_sorted_this[bt_id[0]][:,col_ids_sorted]

    # return in the order of decreasing m200
    sorting = np.argsort(all_nodes_sorted[ids.astype(np.int32),col_w])[::-1]
    ids_sorted = ids[sorting]
    dxy_sorted = dxy[sorting]

      
    return ids,dxy

def connect_node(node_id):

    if len(nodes_used) == 0:
        nodes_used.append(node_id)
        print 'added node % 5d %2.2e' % (node_id,all_nodes_sorted[node_id,col_w])
    else:
        ids, dxy = get_nearest_nodes(node_id)
        dz = np.abs(all_nodes_sorted[ids.astype(np.int32),col_z] - all_nodes_sorted[node_id,col_z]) 

        add_node=True
        for i in range(len(ids)):
            if (ids[i] in nodes_used) & (dxy[i] < min_dist_deg) & (dz[i] < min_dist_z):
                add_node=False
                # print 'skipping node %d as it is close to %d with dx=%2.2f deg' % (node_id, ids[i], dxy[i])
        if add_node:
            nodes_used.append(node_id)
            print 'added node % 5d min(dx)=%2.4f, min(dz)=%2.4f, len(nodes_used)=%d mass=%2.2e' % (node_id,min(dxy),min(dz),len(nodes_used),all_nodes_sorted[node_id,col_w])


def get_graph(all_nodes,min_dist,min_z):
    """
    X :  id, x1, x2, z, w, id_sorted
    """
    n_nodes = all_nodes.shape[0]
    sorting = np.argsort(all_nodes[:,col_w])[::-1]

    global min_dist_deg
    global min_dist_z
    min_dist_deg = min_dist
    min_dist_z = min_z

    global all_nodes_sorted
    all_nodes_sorted=all_nodes[sorting,:]
    all_nodes_sorted[:,col_ids_sorted]=np.arange(0,n_nodes)

    # x,y,z = cosmology.spherical_to_cartesian_with_redshift(all_nodes_sorted[:,col_ra],all_nodes_sorted[:,col_de],all_nodes_sorted[:,col_z])
    x,y,z = cosmology.spherical_to_cartesian_deg(all_nodes_sorted[:,col_ra],all_nodes_sorted[:,col_de],all_nodes_sorted[:,col_z]*0+1)
    global xyz
    xyz=np.concatenate([x[:,None],y[:,None],z[:,None]],axis=1)
    
    for node_id in range(n_nodes):

        connect_node(node_id)

    print 'using %d / %d'  % (len(nodes_used),n_nodes)
    import pylab as pl
    pl.figure()
    pl.scatter(all_nodes_sorted[:,col_ra],all_nodes_sorted[:,col_de],c='b')
    pl.scatter(all_nodes_sorted[nodes_used,col_ra],all_nodes_sorted[nodes_used,col_de],c='r')
    filename_fig  = 'graph_cover.png'
    pl.savefig(filename_fig)

    return all_nodes_sorted[nodes_used,col_id].astype(np.int)



def remove_similar_connections(pairs,min_angle):

    remove_list = []
    pairs_new = range(len(pairs))
    n_pairs = len(pairs)

    n_all=0

    for ic1,vc1 in enumerate(pairs):
        for ic2,vc2 in enumerate(pairs):

            # only one triangle:
            if ic1>ic2:
        
                # connection 1 and 2 have the same node
                same_node=None
                if vc1['ih1'] == vc2['ih1']:
                    same_node_ra , same_node_de = vc2['ra1'] ,  vc2['dec1']
                    other1_ra , other1_de = vc1['ra2'] , vc1['dec2'] 
                    other2_ra , other2_de = vc2['ra2'] , vc2['dec2']
                elif vc1['ih2'] == vc2['ih2']:
                    same_node_ra , same_node_de = vc2['ra2'] ,  vc2['dec2']
                    other1_ra , other1_de = vc1['ra1'] , vc1['dec1'] 
                    other2_ra , other2_de = vc2['ra1'] , vc2['dec1']
                elif vc1['ih1'] == vc2['ih2']:
                    same_node_ra , same_node_de = vc2['ra2'] ,  vc2['dec2']
                    other1_ra , other1_de = vc1['ra2'] , vc1['dec2'] 
                    other2_ra , other2_de = vc2['ra1'] , vc2['dec1']
                elif vc1['ih2'] == vc2['ih1']:
                    same_node_ra , same_node_de = vc2['ra1'] ,  vc2['dec1']
                    other1_ra , other1_de = vc1['ra1'] , vc1['dec1'] 
                    other2_ra , other2_de = vc2['ra2'] , vc2['dec2']

                else:
                    continue

                # decide if to remove connection or not

                x1=np.array( [other1_ra , other1_de] ) - np.array( [ same_node_ra , same_node_de ] )
                x2=np.array( [other2_ra , other2_de] ) - np.array( [ same_node_ra , same_node_de ] )
                angle = np.arccos( np.dot(x1,x2) / np.linalg.norm(x1) / np.linalg.norm(x2) ) /np.pi * 180

                if angle > min_angle:
                    # print '%4d %4d %4d %4d %5.4f  - saved both %d %d' % (vc1['ih1'],vc1['ih2'],vc2['ih1'],vc2['ih2'],angle,vc1['ipair'],vc2['ipair']) 
                    pass
                else:
                    len1= vc1['R_pair'] 
                    len2= vc2['R_pair'] 
                    if len1<=len2: # take shorter connection
                        remove_list.append(vc2['ipair'])
                    else:
                        remove_list.append(vc1['ipair'])
                    n_all+=1
                    if n_all % 100 == 0 : print n_all, n_pairs**2/2., len(remove_list), len(set(remove_list))

    for ip,vp in enumerate(pairs): 
        if vp['ipair'] in remove_list:
            pairs_new.remove(ip)

    return pairs_new
 
