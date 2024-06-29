import torch
from . import sets, ragged
from aljpy import arrdict

def underoccupied(source_idxs, target_idxs, terminal, capacity):
    # Figure out which boxes are oversubscribed and need to be further subdivided
    source_unique, source_counts = torch.unique(source_idxs, return_counts=True)
    target_unique, target_counts = torch.unique(target_idxs, return_counts=True)

    source_okay = torch.ones_like(terminal)
    source_okay[source_unique] = (source_counts <= capacity)
    target_okay = torch.ones_like(terminal)
    target_okay[target_unique] = (target_counts <= capacity)

    return source_okay & target_okay


def orthantree(scaled, capacity=8):
    """Constructs a :ref:`tree <presolve>` for the given :func:`~pybbfmm.scale`'d problem.

    This is a bit of a mess of a function, but long story short it starts with all the sources allocated to the root
    and repeatedly subdivides overfull boxes, constructing the various tree tensors as it goes.

    :param scaled: :func:`~pybbfmm.scale`'d problem.
    :param capacity: the max number of sources or targets per box.
    :return: A :ref:`tree <presolve>`.
    """
    D = scaled.sources.shape[1]

    points = torch.cat([scaled.sources, scaled.targets])
    indices = points.new_zeros((len(points),), dtype=torch.long)

    tree = arrdict.arrdict(
        parents=indices.new_full((1,), -1),
        depths=indices.new_zeros((1,)),
        centers=points.new_zeros((1, D)),
        terminal=indices.new_ones((1,), dtype=torch.bool),
        children=indices.new_full((1,) + (2,)*D, -1),
        descent=indices.new_zeros((1, D)))

    bases = 2**torch.flip(torch.arange(D, device=indices.device), (0,))
    subscript_offsets = sets.cartesian_product(torch.tensor([0, 1], device=indices.device), D)
    center_offsets = sets.cartesian_product(torch.tensor([-1, +1], device=indices.device), D)

    depthcounts = [torch.as_tensor([1], device=indices.device)]

    depth = 0
    while True:
        used, used_inv = torch.unique(indices, return_inverse=True)
        source_idxs, target_idxs = indices[:len(scaled.sources)], indices[-len(scaled.targets):]
        tree.terminal[used] = underoccupied(source_idxs, target_idxs, tree.terminal, capacity)[used] 
        
        used_is_active = ~tree.terminal[used]
        point_is_active = used_is_active[used_inv]
        if not point_is_active.any():
            break

        depth += 1
        
        active = used[used_is_active]
        active_inv = (used_is_active.cumsum(0) - used_is_active.long())[used_inv[point_is_active]]
        first_child = len(tree.parents) + 2**D*torch.arange(len(active), device=active.device)
        point_offset = ((points[point_is_active] >= tree.centers[active][active_inv])*bases).sum(-1)
        child_box = first_child[active_inv] + point_offset
        indices[point_is_active] = child_box

        trailing_ones = (slice(None),) + (None,)*D
        tree.children[active] = first_child[trailing_ones] + (subscript_offsets*bases).sum(-1)

        centers = tree.centers[active][trailing_ones] + center_offsets.float()/2**depth
        descent = center_offsets[None].expand_as(centers)

        n_children = len(active)*2**D
        children = arrdict.arrdict(
            parents=active.repeat_interleave(2**D),
            depths=tree.depths.new_full((n_children,), depth),
            centers=centers.reshape(-1, D),
            descent=descent.reshape(-1, D),
            terminal=tree.terminal.new_ones((n_children,)),
            children=tree.children.new_full((n_children,) + (2,)*D, -1))
        tree = arrdict.cat([tree, children])

        depthcounts.append(n_children)

    tree['id'] = torch.arange(len(tree.parents), device=points.device)

    indices = arrdict.arrdict(
        sources=indices[:len(scaled.sources)],
        targets=indices[-len(scaled.targets):])

    depths = ragged.Ragged(
        torch.arange(len(tree.id), device=points.device),
        torch.as_tensor(depthcounts, device=points.device))

    return tree, indices, depths

def child_boxes(tree, indices, descent):
    subscripts = ((descent + 1)/2).long()
    return tree.children[(indices.long(), *subscripts.T)]

def neighbour_boxes(tree, indices, directions):
    """Finds the neighbour of `indices` in the tree in the given direction.

    There's a non-vectorized (and easier to understand) version of this function 
    `here <https://stackoverflow.com/questions/32412107/quadtree-find-neighbor/61211884#61211884>`_.
    """
    #TODO: This can be framed as a recursive scheme and then as a dynamic programming scheme. 
    # Should save a factor of log(n)
    indices = torch.as_tensor(indices, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = torch.as_tensor(directions, dtype=tree.parents.dtype, device=tree.parents.device)
    directions = directions[None].repeat((len(indices), 1))
    assert len(directions) == len(indices), 'There should be as many directions as indices'

    current = indices.clone()
    alive = [torch.ones_like(indices, dtype=torch.bool)]
    neighbour_descents = []
    while alive[-1].any():
        live = alive[-1] & (directions != 0).any(-1) & (current >= 0)
        alive.append(live)

        descent = tree.descent[current]
        neighbour_descents.append(descent*(1 - 2*directions.abs()))

        directions = (descent + directions).div(2).long() 
        current[live] = tree.parents[current[live]]

    for descent, live in zip(neighbour_descents[::-1], alive[::-1]):
        internal = ~tree.terminal[current] & (current >= 0) & live
        current[internal] = child_boxes(tree, current[internal], descent[internal])

    return current

def u_scheme(tree, neighbours):
    """Calculates the u-:ref:`scheme <presolve>`.
    """
    unique_neighbours = torch.sort(neighbours, 1, descending=True).values
    unique_neighbours[:, 1:][unique_neighbours[:, 1:] == unique_neighbours[:, :-1]] = -1

    pairs = torch.stack([tree.id[:, None].expand_as(neighbours), unique_neighbours], -1)
    pairs = pairs[(pairs >= 0).all(-1) & tree.terminal[pairs].all(-1)]

    partner_is_larger = tree.depths[pairs[:, 0]] > tree.depths[pairs[:, 1]]
    smaller_partners = torch.flip(pairs[partner_is_larger], (1,))
    pairs = torch.cat([pairs, smaller_partners])

    return ragged.from_pairs(pairs, len(tree.id), len(tree.id))

def v_scheme(tree, depths, directions, neighbours):
    """Calculates the v-:ref:`scheme <presolve>`.
    """
    D = tree.children.ndim-1
    nonzero_directions = (directions != 0).any(-1)
    descents = sets.flat_cartesian_product(torch.tensor([-1, +1], device=tree.id.device), D)

    # The v list is many times bigger than the other lists, so we'll loop rather than 
    # vectorize to preserve memory.
    result = []
    for d in nonzero_directions.nonzero().squeeze(1):
        colleagues = neighbours[tree.parents, d]
        for friend_descent in descents:
            friends = child_boxes(tree, colleagues, friend_descent)
            for own_descent in descents:
                offset = (-own_descent + 4*directions[d] + friend_descent)/2
                if (offset.abs() <= 1).all(-1):
                    continue

                for depth in torch.arange(depths.domain, device=tree.id.device):
                    s = depths.slice(depth)
                    mask = (tree.descent[s] == own_descent).all(-1) & ~tree.terminal[colleagues[s]] & (colleagues[s] >= 0)
                    result.append(arrdict.arrdict(
                        boxes=tree.id[s][mask],
                        friends=friends[s][mask],
                        offset=offset,
                        depth=depth)) 

    return result

def w_pairs(tree, directions, neighbours):
    """Metadata needed for calculating w- and x-:ref:`scheme <presolve>`.
    """
    D = tree.children.ndim-1
    bs = tree.terminal.nonzero().squeeze(1)

    origins, colleagues, dirs = [], [], []
    for d, dns in zip(directions, neighbours.T):
        dns = dns[bs]
        is_colleague = (tree.depths[bs] == tree.depths[dns])
        valid = is_colleague & ~tree.terminal[dns]
        origins.append(bs[valid])
        colleagues.append(dns[valid])
        dirs.append(d[None].repeat_interleave(valid.sum(), 0))
    origins, colleagues, dirs = torch.cat(origins), torch.cat(colleagues), torch.cat(dirs, 0)

    pairs = [origins.new_empty((0, 2))]
    parents = colleagues
    while parents.nelement():
        friends = tree.children[parents].reshape(-1, 2**D)
        distant = (tree.descent[friends] == dirs[:, None, :]).any(-1)
        
        ps = torch.stack([origins[:, None].expand_as(friends), friends], -1)
        pairs.append(ps[distant])
        
        mask = ~distant & ~tree.terminal[friends]
        origins, parents = ps[mask].T
        dirs = dirs[:, None].repeat_interleave(2**D, 1)[mask]
    pairs = torch.cat(pairs)

    return pairs

def interaction_scheme(tree, depths):
    """Returns the datastructures needed to calculate the :ref:`interactions <presolve>` between boxes.
    
    The datastructures are pretty heterogeneous because, well, performance. They're set
    up so the data needed can be got at fast without blowing up the memory budget.

    :param tree: a :ref:`tree <presolve>`.
    :param depths: the :ref:`depths <presolve>` to go with the tree.
    :return: a :ref:`scheme <presolve>`.
    """
    D = tree.children.ndim-1
    directions = sets.flat_cartesian_product(torch.tensor([-1, 0, +1], device=tree.id.device), D)
    neighbours = torch.stack([neighbour_boxes(tree, tree.id, d) for d in directions], -1)

    w = w_pairs(tree, directions, neighbours)

    return arrdict.arrdict(
        w=ragged.from_pairs(w, len(tree.id), len(tree.id)),
        x=ragged.from_pairs(w.flip((1,)), len(tree.id), len(tree.id)),
        u=u_scheme(tree, neighbours),
        v=v_scheme(tree, depths, directions, neighbours))