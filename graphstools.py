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


def get_graph_topo(halos):

    x,y,z = cosmology.spherical_to_cartesian_with_redshift(halos['ra'],halos['dec'],halos['z'])
    box_coords = np.concatenate( [x,y,z] , axis=1)
    BT = BallTree(box_coords, leaf_size=5)
    list_conn = []
    for ih,vh in enumerate(halos):
       
        n_connections=70
        bt_dx,bt_id = BT.query(box_coords[ih,:],k=n_connections)

        for ic,vc in enumerate(halos[bt_id]):

            pass
            


        # DS=comology.get_angular_separation(vh['ra'],vh['dec'],halos['ra'],halos['dec'])
        # DA=comology.get_ang_diam_dist(vh['z'],halos['z'])




def get_triangulation(pairs,halo1,halo2,halos):

    select = halos['m200_fit']>5e13
    halos=halos[select].copy()
    find_neighbors = lambda x,triang: list(set(indx for simplex in triang.simplices if x in simplex for indx in simplex if indx !=x))
    # find_neighbors = lambda pindex,triang: triang.vertex_neighbor_vertices[1][triang.vertex_neighbor_vertices[0][pindex]:triang.vertex_neighbor_vertices[0][pindex+1]]
    import scipy.spatial
    # coords = np.concatenate([halos['ra'][:,None],halos['dec'][:,None],halos['z'][:,None]],axis=1)
    coords = np.concatenate([halos['ra'][:,None],halos['dec'][:,None]],axis=1)
    dt=scipy.spatial.Delaunay(coords)
    list_ne = []
    for i in range(dt.simplices.shape[0]):
        v=[ dt.simplices[i][0] , dt.simplices[i][1] ];
        if v not in list_ne: list_ne.append(v)
        v=[ dt.simplices[i][0] , dt.simplices[i][2] ];
        if v not in list_ne: list_ne.append(v)
        # v=[ dt.simplices[i][0] , dt.simplices[i][3] ];
        if v not in list_ne: list_ne.append(v)
        v=[ dt.simplices[i][1] , dt.simplices[i][2] ];
        if v not in list_ne: list_ne.append(v)
        # v=[ dt.simplices[i][1] , dt.simplices[i][3] ];
        if v not in list_ne: list_ne.append(v)
        # v=[ dt.simplices[i][2] , dt.simplices[i][3] ];
        if v not in list_ne: list_ne.append(v)
        
        print i

    ne=np.array(list_ne)    
    import pylab as pl
    pl.figure()
    pl.plot([halos['ra'][ne[:,0]],halos['ra'][ne[:,1]]],[halos['dec'][ne[:,0]],halos['dec'][ne[:,1]]],c='r')

    import tabletools
    tabletools.savePickle('trian.pp2',ne)
    import pdb; pdb.set_trace()
    import pdb; pdb.set_trace()

def get_best_neighbour(pairs,halo1,halo2):

    select_list = []
    used_nodes = []
    Dtot = np.sqrt(pairs['Dxy']**2 + pairs['Dlos']**2)
    # select_cut = ( (halo1['m200_fit'] > 1e14) & (halo2['m200_fit'] > 1e13) & (Dtot < 11) & (Dtot > 5) & (pairs['Dlos'] > 4) & (pairs['Dxy'] > 5) & ((halo1['m200_sig'] > 0) | (halo2['m200_sig'] > 0)) & (halo2['m200_sig'] > 0) )  # 001-lrgs 
    # select_cut = ( (halo1['m200_fit'] > 1e13) & (halo2['m200_fit'] > 1e13) & (Dtot < 11) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 4) & ((halo1['m200_sig'] > 2) | (halo2['m200_sig'] > 2)) ) # 009-lrgs
    # select_cut = ( (halo1['m200_fit'] > 1e13) & (halo2['m200_fit'] > 1e13) & (Dtot < 12) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 4) & ((halo1['m200_sig'] > 2) | (halo2['m200_sig'] > 2)) ) # 010-lrgs-topo 2.02 35
    # select_cut = ( (halo1['m200_fit'] > 1e13) & (halo2['m200_fit'] > 1e13) & (Dtot < 22) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 6) & ((halo1['m200_sig'] > 2.) | (halo2['m200_sig'] > 2.)) ) # 011-kappaK 48  4.65  
    # select_cut = ( (halo1['m200_fit'] < 1.5e14) & (halo2['m200_fit'] > 1e13) & (Dtot < 22) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 6) & ((halo1['m200_sig'] > 2.) | (halo2['m200_sig'] > 2.)) ) # 011-kappaK 48  4.65  

    # 47 pairs - main result
    # select_cut = ( ((halo1['m200_fit'] > 1e13) | (halo2['m200_fit'] > 1e13)) & (Dtot < 15) & (Dtot > 6) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 5) & ((halo1['m200_sig'] > 1.75) | (halo2['m200_sig'] > 1.75)) ) # 011-kappaK 48  4.65  

    # new fix
    # select_cut = ( ((halo1['m200_fit'] > 1.e13) | (halo2['m200_fit'] > 1.e13)) & (Dtot < 15) & (Dtot > 6) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 5) & ((halo1['m200_sig'] > 2.2) | (halo2['m200_sig'] > 2.2)) ) 
    # select_cut = ( ((halo1['m200_fit'] > 1.5e14) | (halo2['m200_fit'] > 1.5e14)) & (Dtot < 15) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 5) & ((halo1['m200_sig'] > 1.5) | (halo2['m200_sig'] > 1.5)) ) 
    # select_cut = ( ((halo1['m200_fit'] > 1.5e14) | (halo2['m200_fit'] > 1.5e14)) & (Dtot < 15) & (Dtot > 6) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 5) & ((halo1['m200_sig'] > 2.) | (halo2['m200_sig'] > 2.)) ) 

    # 19 pairs
    select_cut = ( ((halo1['m200_fit'] > 2.5e14) | (halo2['m200_fit'] > 2.5e14)) & (halo2['m200_fit'] > 1e13) & (halo1['m200_fit'] > 1e13) & (Dtot < 12) & (Dtot > 4) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 4) & ((halo1['m200_sig'] > 1.) & (halo2['m200_sig'] > 1.)) ) 

    # large Rpair
    # select_cut = ( ((halo1['m200_fit'] > 2.5e14) | (halo2['m200_fit'] > 2.5e14)) & (halo2['m200_fit'] > 1e13) & (halo1['m200_fit'] > 1e13) & (Dtot < 20) & (Dtot > 4) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 4) & ((halo1['m200_sig'] > 1.) & (halo2['m200_sig'] > 1.)) ) 
    # select_cut = ( ((halo1['m200_fit'] > 2.5e14) | (halo2['m200_fit'] > 2.5e14)) & (halo2['m200_fit'] > 1e13) & (halo1['m200_fit'] > 1e13) & (Dtot < 20) & (Dtot > 8) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 8) & ((halo1['m200_sig'] > 1.) & (halo2['m200_sig'] > 1.)) ) 
    select_list = np.nonzero(select_cut)[0].tolist()
    return select_list
   
    # select_cut = ( ((halo1['m200_fit'] > 1e13) | (halo2['m200_fit'] > 1e13)) & (Dtot < 15) & (Dtot > 6) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 8) & ((halo1['m200_sig'] > 1.) | (halo2['m200_sig'] > 1.)) ) # 011-kappaK 48  4.65  
    
    # random
    # select_cut = ( (halo1['m200_fit'] > 1e13) & (halo2['m200_fit'] > 1e13) & (Dtot < 22) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 6)  ) 

    # select_cut = ( (halo1['m200_fit'] > 1e13) & (halo2['m200_fit'] > 1e13) & (Dtot < 1000) & (Dtot > 0) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 0) & ((halo1['m200_sig'] > 0) | (halo2['m200_sig'] > 0)) ) # 010-lrgs-topo 2.05 37
    # select_cut = ( (halo1['m200_fit'] > 1e13) & (halo2['m200_fit'] > 1e13) & (Dtot < 12) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 4) & ((halo1['m200_sig'] > 2) | (halo2['m200_sig'] > 2)) ) # 010-lrgs-topo
    # select_cut = ( (halo1['m200_fit'] > 1e13) & (halo2['m200_fit'] > 1e13) & (Dtot < 11) & (Dtot > 5) & (pairs['Dlos'] > 4) & (pairs['Dxy'] > 4) & ((halo1['m200_sig'] > 2) | (halo2['m200_sig'] > 2)) & (halo2['m200_sig'] > 0) ) 
    # select_cut = ( (halo1['m200_fit'] > 1e14) & (halo2['m200_fit'] > 1e13) & (Dtot < 11) & (Dtot > 5) & (pairs['Dlos'] > 0) & (pairs['Dxy'] > 5) & ((halo1['m200_sig'] > 2) | (halo2['m200_sig'] > 0)) & (halo2['m200_sig'] > 0) ) 

    for ip,vp in enumerate(pairs):
        add=False
        select = ((pairs['ih1'] == vp['ih1']) | (pairs['ih2'] == vp['ih1'])) & select_cut
        d =np.sqrt(pairs[select]['Dxy']**2 + pairs[select]['Dlos']**2)**(1/2.)
        # d = 1/pairs[select]['Dlos']
        # q =  (halo2[select]['m200_fit'])/1e14/d
        q =  (halo2[select]['m200_fit'])/1e14
        # q =  1./d

        dist1 = cosmology.get_angular_separation(vp['ra1'],vp['dec1'],pairs[select_list]['ra1'],pairs[select_list]['dec1'],unit='deg')
        dist2 = cosmology.get_angular_separation(vp['ra1'],vp['dec1'],pairs[select_list]['ra2'],pairs[select_list]['dec2'],unit='deg')
        dist3 = cosmology.get_angular_separation(vp['ra2'],vp['dec2'],pairs[select_list]['ra1'],pairs[select_list]['dec1'],unit='deg')
        dist4 = cosmology.get_angular_separation(vp['ra2'],vp['dec2'],pairs[select_list]['ra2'],pairs[select_list]['dec2'],unit='deg')
        dist = np.concatenate([dist1,dist2,dist3,dist4])

        if np.any(dist < 0.1):
            continue

        if len(np.nonzero(select)[0]) == 0:
            continue    
        else:
            # q =  halo2[select]['m200_fit']

            sorting = q.argsort()[::-1]
            id_best = pairs[select]['ipair'][sorting[0]]
            if id_best != ip:
                continue
            # if pairs[id_best]['eyeball_class']==0:
            #     import pdb; pdb.set_trace()
            if (id_best not in select_list) & (vp['ih1'] not in used_nodes) & (vp['ih2'] not in used_nodes): 

                select_list.append(id_best)
                used_nodes.append(vp['ih1'])
                used_nodes.append(vp['ih2'])
                add=True

        if add:
            str_ids = ' '.join([ '%2d'%i for i in pairs[select]['ipair']])
            str_mass = ' '.join([ '%2.1e'%i for i in halo2[select]['m200_fit']])
            str_qs = ' '.join([ '%2.2f'%i for i in q])
            str_ds = ' '.join([ '%2.2f'%i for i in d])
            str_nodes = str([pairs[i]['eyeball_class'] for i in np.nonzero(select)[0]])
            str_class = sum(pairs[select_list]['eyeball_class']==0), sum(pairs[select_list]['eyeball_class']==1), sum(pairs[select_list]['eyeball_class']==2)
            print '% 3d\t% 5.2f\t% 5.2f\t%20s\t%20s\t%30s\t%40s\t%40s\t%20s' % (ip,vp['ra1'],vp['dec1'],str_ids,str_nodes,str_mass,str_qs,str_ds,str_class)


    # for ip in select_list: 
        # if select_cut2[ip]==False: select_list.remove(ip)


    return select_list

def get_clean_connections(lrg_pairs,clusters):

    select=10**clusters['m200'] > 1e14
    clusters=clusters.copy()[select]

    # init to false
    select_clean=lrg_pairs['ipair'] < -1
    radius_frac=0.
    min_dz=0.1
    min_dz_lrg=0.01

    for ip,vp in enumerate(lrg_pairs):
        clus_dist=cosmology.get_angular_separation(vp['ra_mid'],vp['dec_mid'],clusters['ra'],clusters['dec'])
        radius=cosmology.get_angular_separation(vp['ra1'],vp['dec1'],vp['ra2'],vp['dec2'])*radius_frac/2
        select1 = (clus_dist<radius) & (np.abs(vp['z']-clusters['z'])<min_dz)

        lrgs_ra = np.concatenate([lrg_pairs['ra1'],lrg_pairs['ra2']])
        lrgs_de = np.concatenate([lrg_pairs['dec1'],lrg_pairs['dec2']])
        lrgs_z = np.concatenate([lrg_pairs['z'],lrg_pairs['z']])
        
        clus_dist=cosmology.get_angular_separation(vp['ra_mid'],vp['dec_mid'],lrgs_ra,lrgs_de)
        radius=cosmology.get_angular_separation(vp['ra1'],vp['dec1'],vp['ra2'],vp['dec2'])*radius_frac/2
        select2 = (clus_dist<radius) & (np.abs(vp['z']-lrgs_z)<min_dz_lrg)

        if ( (~np.any(select1)) & (~np.any(select2)) ):
            select_clean[ip] = True

    return select_clean



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
 
