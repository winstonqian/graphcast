import numpy as np
import gsd.hoomd



def ase_traj_to_gsd(vertices, edges, save_path, num_levels, size_scale=1.2):

    with gsd.hoomd.open(save_path, "w") as hoomd_traj:
        pos = vertices
        gsd_frame = gsd.hoomd.Frame()
        # gsd_frame.configuration.step = step

        gsd_frame.particles.N = pos.shape[0]
        gsd_frame.particles.position = pos
        num_types = num_levels
        gsd_frame.particles.types = [str(i) for i in range(num_types)]
        print("len(vertices): ", len(vertices))
        num_vertices_per_level = int(len(vertices) / num_types)

        type_ids = np.array([i for i in range(num_types) for _ in range(num_vertices_per_level)], dtype=np.int32)
        gsd_frame.particles.typeid = type_ids
        # gsd_frame.particles.typeid = np.array([0] * 2562 + [1] * 2562, dtype=np.int32)
        # gsd_frame.particles.diameter = [
        #     covalent_radii[atom_types[i]] * size_scale
        #     for i in range(len(atom_types))
        # ]

        gsd_frame.bonds.N = len(edges)
        gsd_frame.bonds.types = [str(i) for i in range(num_types+1)]
        num_inter_mesh_edges = num_vertices_per_level * (num_types-1)
        # Generate edge labels
        labels = [str(i[0] // num_vertices_per_level) for i in edges[:-num_inter_mesh_edges]]

        # Output to gsd_frame.bonds.typeid
        gsd_frame.bonds.typeid = np.array(labels + [4] * num_inter_mesh_edges, dtype=str)
        print("len(edges)", len(edges))
        print("len(labels)", len(labels))
        print("len(gsd_frame.bonds.typeid)", len(gsd_frame.bonds.typeid))
                # gsd_frame.bonds.types = ['0', '1', '2']
        # gsd_frame.bonds.typeid = np.array([0] * 240 + [1] * 240 + [2] * 42, dtype=np.int32)
        gsd_frame.bonds.group = edges
        '''Combine sender and receiver'''

        cell_length = np.array(
            [10, 10, 10]
        ) 
        '''Make it large, e.g. 100'''
        # setting angles to 90 for now
        cell_angle = np.empty_like(cell_length)
        cell_angle.fill(90)
        gsd_frame.configuration.box = lengths_and_angles_to_tilt_factors(
            cell_length[0],
            cell_length[1],
            cell_length[2],
            cell_angle[0],
            cell_angle[1],
            cell_angle[2],
        )
        hoomd_traj.append(gsd_frame)

def lengths_and_angles_to_tilt_factors(
    a_length,
    b_length,
    c_length,
    alpha,
    beta,
    gamma,
):
    """
    Copied from: https://github.com/mdtraj/mdtraj/blob/main/mdtraj/utils/unitcell.py#L180
    Parameters
    ----------
    a_length : scalar or np.ndarray
        length of Bravais unit vector **a**
    b_length : scalar or np.ndarray
        length of Bravais unit vector **b**
    c_length : scalar or np.ndarray
        length of Bravais unit vector **c**
    alpha : scalar or np.ndarray
        angle between vectors **b** and **c**, in degrees.
    beta : scalar or np.ndarray
        angle between vectors **c** and **a**, in degrees.
    gamma : scalar or np.ndarray
        angle between vectors **a** and **b**, in degrees.

    Returns
    -------
    lx : scalar
        Extent in x direction
    ly : scalar
        Extent in y direction
    lz : scalar
        Extent in z direction
    xy : scalar
        Unit vector **b** tilt with respect to **a**
    xz : scalar
        Unit vector of **c** tilt with respect to **a**
    yz : scalar
        Unit vector of **c** tilt with respect to **b**
    """
    lx = a_length
    xy = b_length * np.cos(np.deg2rad(gamma))
    xz = c_length * np.cos(np.deg2rad(beta))
    ly = np.sqrt(b_length**2 - xy**2)
    yz = (b_length * c_length * np.cos(np.deg2rad(alpha)) - xy * xz) / ly
    lz = np.sqrt(c_length**2 - xz**2 - yz**2)

    return np.array([lx, ly, lz, xy, xz, yz])