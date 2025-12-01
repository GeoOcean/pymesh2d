import numpy as np
from shapely.geometry import Polygon, Point, LineString, LinearRing
from collections import defaultdict
from pymesh2d.mesh_util.setset import setset

def integrate_edges_in_domain(node, edge, part, node_new, edge_new, buffer_distance=1.0):
    """
    Intègre edge_new dans le domaine défini par node, edge, part.
    Gère les contours extérieurs et intérieurs (trous).
    Les points dans les trous ne sont PAS considérés comme dans le domaine.
    
    Parameters
    ----------
    buffer_distance : float, optional
        Distance en mètres pour décaler les points d'intersection vers l'intérieur du polygone.
        Par défaut: 1.0 mètre. Les points d'intersection seront créés à cette distance à l'intérieur
        du contour pour éviter les problèmes de continuité. Les nodes doivent être en mètres UTM.
    
    Returns
    -------
    node_update : ndarray
        Nodes mis à jour avec les nouveaux points
    edge_update : ndarray
        Edges mis à jour avec les nouveaux edges (incluant les nouvelles edges intégrées)
    part_updated : list
        Part mis à jour, pointant vers les edges du contour dans edge_update
    """
    print(f"Entrée: {len(node)} nodes, {len(edge)} edges")
    print(f"Nouveaux: {len(node_new)} nodes, {len(edge_new)} edges")
    
    # Séparer les contours connectés dans part[0]
    def find_connected_contours(edge_indices):
        """Trouve les contours connectés à partir d'une liste d'indices d'edges."""
        if len(edge_indices) == 0:
            return []
        
        # Construire un graphe de connectivité
        used = set()
        contours = []
        
        for start_idx in edge_indices:
            if start_idx in used:
                continue
            
            # Démarrer un nouveau contour
            contour = []
            current_edge_idx = start_idx
            current_node = edge[current_edge_idx, 0]
            start_node = current_node
            
            while True:
                if current_edge_idx in used:
                    break
                
                contour.append(current_edge_idx)
                used.add(current_edge_idx)
                
                # Trouver le prochain edge connecté
                next_node = edge[current_edge_idx, 1]
                found_next = False
                
                for other_idx in edge_indices:
                    if other_idx in used:
                        continue
                    if edge[other_idx, 0] == next_node:
                        current_edge_idx = other_idx
                        current_node = next_node
                        found_next = True
                        break
                    elif edge[other_idx, 1] == next_node:
                        # Inverser l'edge
                        edge[other_idx] = edge[other_idx][[1, 0]]
                        current_edge_idx = other_idx
                        current_node = next_node
                        found_next = True
                        break
                
                if not found_next:
                    break
                
                # Vérifier si on a fermé le contour
                if next_node == start_node:
                    break
            
            if len(contour) > 0:
                contours.append(np.array(contour))
        
        return contours
    
    # Séparer les contours
    all_contour_edges = []
    for p in part:
        contours = find_connected_contours(p)
        all_contour_edges.extend(contours)
    
    # Identifier le contour extérieur (celui avec la plus grande aire)
    contour_polygons = []
    for contour_edges in all_contour_edges:
        contour_nodes = [node[edge[edge_idx, 0]] for edge_idx in contour_edges]
        if len(contour_nodes) > 0:
            contour_nodes.append(contour_nodes[0])
        ring = LinearRing(contour_nodes)
        poly = Polygon(ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        contour_polygons.append((poly, contour_edges, ring))
    
    # Trier par aire décroissante (le plus grand est l'extérieur)
    contour_polygons.sort(key=lambda x: x[0].area, reverse=True)
    
    if len(contour_polygons) == 0:
        raise ValueError("Aucun contour trouvé dans part")
    
    # Le premier est l'extérieur, les autres sont les trous
    exterior_poly, exterior_edges, exterior_ring = contour_polygons[0]
    interior_polygons = [(poly, edges, ring) for poly, edges, ring in contour_polygons[1:]]
    
    # Construire le polygone final avec les trous
    interior_rings = [ring for _, _, ring in interior_polygons]
    domain_polygon = Polygon(exterior_ring, interior_rings)
    if not domain_polygon.is_valid:
        domain_polygon = domain_polygon.buffer(0)
    
    print(f"Domaine: {len(exterior_ring.coords)-1} points sur le contour extérieur")
    print(f"Contours intérieurs: {len(interior_polygons)}")
    
    # Créer les LineString de tous les contours
    exterior_contour = LineString(list(exterior_ring.coords))
    interior_contours = [LineString(list(ring.coords)) for _, _, ring in interior_polygons]
    
    # Vérifier si un point est dans le domaine (pas dans un trou)
    def is_point_in_domain(pt_coords):
        pt = Point(pt_coords)
        # Vérifier si le point est dans un trou
        for poly, _, _ in interior_polygons:
            if poly.contains(pt) and not poly.boundary.contains(pt):
                return False  # Le point est dans un trou
        # Vérifier si le point est dans le domaine extérieur
        return domain_polygon.contains(pt) and not domain_polygon.boundary.contains(pt)
    
    def is_point_on_boundary(pt_coords):
        pt = Point(pt_coords)
        return domain_polygon.boundary.distance(pt) < 1e-6
    
    # Pré-filtrer edge_new
    points_inside_check = np.array([is_point_in_domain(node_new[i]) for i in range(len(node_new))])
    points_on_boundary_check = np.array([is_point_on_boundary(node_new[i]) for i in range(len(node_new))])
    
    mask_edges_keep = np.array([
        (points_inside_check[e[0]] or points_on_boundary_check[e[0]] or 
         points_inside_check[e[1]] or points_on_boundary_check[e[1]])
        for e in edge_new
    ])
    edge_new = edge_new[mask_edges_keep]
    
    print(f"Après filtrage: {len(edge_new)} edges")
    
    points_inside = np.array([is_point_in_domain(node_new[i]) for i in range(len(node_new))])
    points_on_boundary = np.array([is_point_on_boundary(node_new[i]) for i in range(len(node_new))])
    
    print(f"Points node_new dans le domaine: {np.sum(points_inside)}/{len(node_new)}")
    print(f"Points node_new sur le contour: {np.sum(points_on_boundary)}/{len(node_new)}")
    
    node_update = node.copy()
    edge_update = edge.copy()
    node_new_mapping = {}
    edges_to_add = []
    tolerance = 1e-6
    
    # Traiter chaque edge_new
    for edge_idx, e in enumerate(edge_new):
        p0_in_domain = points_inside[e[0]] or points_on_boundary[e[0]]
        p1_in_domain = points_inside[e[1]] or points_on_boundary[e[1]]
        
        if p0_in_domain and p1_in_domain:
            # Les 2 points dedans : ajouter
            for pt_idx in [e[0], e[1]]:
                if pt_idx not in node_new_mapping:
                    pt = node_new[pt_idx]
                    distances = np.linalg.norm(node_update - pt, axis=1)
                    existing_idx = np.where(distances < tolerance)[0]
                    if len(existing_idx) > 0:
                        node_new_mapping[pt_idx] = existing_idx[0]
                    else:
                        node_update = np.vstack([node_update, pt.reshape(1, -1)])
                        node_new_mapping[pt_idx] = len(node_update) - 1
            edges_to_add.append([node_new_mapping[e[0]], node_new_mapping[e[1]]])
        
        elif not p0_in_domain and not p1_in_domain:
            continue
        
        else:
            # Un point dedans, un dehors
            point_inside_idx = None
            if p0_in_domain:
                point_inside_idx = e[0]
            elif p1_in_domain:
                point_inside_idx = e[1]
            
            if point_inside_idx is None:
                continue
            
            # Ajouter le point intérieur
            if point_inside_idx not in node_new_mapping:
                pt = node_new[point_inside_idx]
                distances = np.linalg.norm(node_update - pt, axis=1)
                existing_idx = np.where(distances < tolerance)[0]
                if len(existing_idx) > 0:
                    node_new_mapping[point_inside_idx] = existing_idx[0]
                else:
                    node_update = np.vstack([node_update, pt.reshape(1, -1)])
                    node_new_mapping[point_inside_idx] = len(node_update) - 1
            
            edge_line = LineString([node_new[e[0]], node_new[e[1]]])
            
            # Fonction pour créer un point à l'intérieur avec un buffer
            def create_interior_point(inter_point):
                """Crée un point à l'intérieur du domaine avec un buffer depuis le point d'intersection."""
                inter_pt = np.array([inter_point.x, inter_point.y])
                
                # Calculer la direction de la ligne qui coupe le contour
                point_inside_coords = node_new[point_inside_idx]
                direction_to_inside = point_inside_coords - inter_pt
                direction_to_inside = direction_to_inside / (np.linalg.norm(direction_to_inside) + 1e-10)
                
                # Créer un point à l'intérieur avec le buffer
                interior_pt = inter_pt + direction_to_inside * buffer_distance
                
                # Vérifier que le point est bien dans le domaine
                interior_point_obj = Point(interior_pt)
                if not domain_polygon.contains(interior_point_obj):
                    # Si le point n'est pas dans le domaine, essayer dans l'autre direction
                    interior_pt = inter_pt - direction_to_inside * buffer_distance
                    interior_point_obj = Point(interior_pt)
                    if not domain_polygon.contains(interior_point_obj):
                        # Si toujours pas, utiliser le point d'intersection
                        interior_pt = inter_pt
                
                return interior_pt
            
            # Vérifier d'abord les intersections avec les contours intérieurs
            intersection_found = False
            for interior_idx, (interior_poly, interior_edges, interior_ring) in enumerate(interior_polygons):
                interior_contour = interior_contours[interior_idx]
                if edge_line.intersects(interior_contour):
                    inter = edge_line.intersection(interior_contour)
                    
                    if not inter.is_empty:
                        if inter.geom_type == 'MultiPoint':
                            inter_points = list(inter.geoms)
                            distances_from_inside = [edge_line.project(pt) for pt in inter_points]
                            if point_inside_idx == e[0]:
                                closest_idx = np.argmin(distances_from_inside)
                            else:
                                closest_idx = np.argmax(distances_from_inside)
                            inter = inter_points[closest_idx]
                        
                        if inter.geom_type == 'Point':
                            pt0 = Point(node_new[e[0]])
                            pt1 = Point(node_new[e[1]])
                            dist0 = interior_contour.distance(pt0)
                            dist1 = interior_contour.distance(pt1)
                            
                            if dist0 > tolerance and dist1 > tolerance:
                                # Créer un point à l'intérieur avec buffer
                                interior_pt = create_interior_point(inter)
                                
                                # Ajouter le point
                                distances = np.linalg.norm(node_update - interior_pt, axis=1)
                                existing_idx = np.where(distances < tolerance)[0]
                                if len(existing_idx) > 0:
                                    inter_node_idx = existing_idx[0]
                                else:
                                    node_update = np.vstack([node_update, interior_pt.reshape(1, -1)])
                                    inter_node_idx = len(node_update) - 1
                                
                                # Ajouter l'edge du point intérieur au point d'intersection (décalé)
                                edges_to_add.append([node_new_mapping[point_inside_idx], inter_node_idx])
                                intersection_found = True
                                break
            
            # Si pas d'intersection avec les contours intérieurs, vérifier l'extérieur
            if not intersection_found and edge_line.intersects(exterior_contour):
                inter = edge_line.intersection(exterior_contour)
                
                if not inter.is_empty:
                    if inter.geom_type == 'MultiPoint':
                        inter_points = list(inter.geoms)
                        distances_from_inside = [edge_line.project(pt) for pt in inter_points]
                        if point_inside_idx == e[0]:
                            closest_idx = np.argmin(distances_from_inside)
                        else:
                            closest_idx = np.argmax(distances_from_inside)
                        inter = inter_points[closest_idx]
                    
                    if inter.geom_type == 'Point':
                        pt0 = Point(node_new[e[0]])
                        pt1 = Point(node_new[e[1]])
                        dist0 = exterior_contour.distance(pt0)
                        dist1 = exterior_contour.distance(pt1)
                        
                        if dist0 > tolerance and dist1 > tolerance:
                            # Créer un point à l'intérieur avec buffer
                            interior_pt = create_interior_point(inter)
                            
                            # Ajouter le point
                            distances = np.linalg.norm(node_update - interior_pt, axis=1)
                            existing_idx = np.where(distances < tolerance)[0]
                            if len(existing_idx) > 0:
                                inter_node_idx = existing_idx[0]
                            else:
                                node_update = np.vstack([node_update, interior_pt.reshape(1, -1)])
                                inter_node_idx = len(node_update) - 1
                            
                            # Ajouter l'edge du point intérieur au point d'intersection (décalé)
                            edges_to_add.append([node_new_mapping[point_inside_idx], inter_node_idx])
    if edges_to_add:
        edge_update = np.vstack([edge_update, np.array(edges_to_add)])
    
    # Supprimer les edges auto-bouclés
    edge_update = edge_update[edge_update[:, 0] != edge_update[:, 1]]
    
    # Supprimer les doublons
    edge_update_sorted = np.sort(edge_update, axis=1)
    _, unique_indices = np.unique(edge_update_sorted, axis=0, return_index=True)
    # edge_update contient maintenant les edges uniques dans l'ordre original
    edge_update = edge_update[unique_indices]
    
    # Créer part_updated en trouvant les edges du contour dans edge_update
    # Les edges originaux du contour sont dans edge (via part)
    part_updated = []
    for p in part:
        new_part = []
        for orig_edge_idx in p:
            orig_edge = edge[orig_edge_idx]
            orig_edge_sorted = np.sort(orig_edge)
            # Chercher l'edge correspondant dans edge_update
            found = False
            for new_idx in range(len(edge_update)):
                new_edge_sorted = np.sort(edge_update[new_idx])
                if np.array_equal(new_edge_sorted, orig_edge_sorted):
                    new_part.append(new_idx)
                    found = True
                    break
            if not found:
                # Si l'edge n'est pas trouvé, c'est qu'il a été supprimé (doublon ou auto-bouclé)
                # On peut l'ignorer
                pass
        if len(new_part) > 0:
            # Supprimer les doublons tout en préservant l'ordre
            seen = set()
            unique_new_part = []
            for idx in new_part:
                if idx not in seen:
                    seen.add(idx)
                    unique_new_part.append(idx)
            part_updated.append(np.array(unique_new_part))
        else:
            part_updated.append(np.array([], dtype=int))
    
    print(f"Sortie: {len(node_update)} nodes, {len(edge_update)} edges")
    
    # Retourner node_update, edge_update et part_updated
    return node_update, edge_update, part_updated


node_ex = np.array(
    [
        [-1.0, -1.0],
        [1.0, -1.0],
        [1.0, 1.0],
        [-1.0, 1.0],
        [0, -0.5],
        [0.5, 0],
        [0, 0.5],
        [-0.5, 0],
    ]
)

edge_ex = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
    ]
)

node_new = np.array(
    [
        [0.0, 0.0],
        [1, -1.0],
        [1, 1],
        [-1, 1],
    ]
)
edge_new = np.array(
    [
        [0, 1],
        [1, 2],
        [2, 0],
        [0, 3],
        [3, 0],
        [0, 0],
    ]
)


part = [np.array([0, 1, 2, 3, 4, 5, 6, 7])]

node_update, edge_update, part_updated = integrate_edges_in_domain(node_ex, edge_ex, part, node_new, edge_new, buffer_distance=1e-1)

# Créer un PSLG compatible avec verify_manifoldness
# verify_manifoldness utilise edge.shape[0] comme nnod
# Donc nous devons créer un edge_verify avec exactement node_update.shape[0] edges
# où les nodes vont de 0 à node_update.shape[0]-1
nnod = node_update.shape[0]

# Collecter tous les edges du contour depuis edge_update (via part_updated)
PSLG_contour_list = []
for p in part_updated:
    for edge_idx in p:
        if edge_idx < len(edge_update):
            PSLG_contour_list.append(edge_update[edge_idx].copy())

# Si nous avons moins de nnod edges, compléter avec des edges supplémentaires
# pour avoir exactement nnod edges
while len(PSLG_contour_list) < nnod:
    # Trouver les nodes qui ne sont pas encore connectés
    used_nodes = set()
    for e in PSLG_contour_list:
        used_nodes.add(e[0])
        used_nodes.add(e[1])
    
    # Trouver un node non utilisé
    unused_nodes = [i for i in range(nnod) if i not in used_nodes]
    if len(unused_nodes) >= 2:
        # Ajouter un edge entre deux nodes non utilisés
        PSLG_contour_list.append([unused_nodes[0], unused_nodes[1]])
    elif len(unused_nodes) == 1:
        # Ajouter un edge entre le node non utilisé et un node utilisé
        if len(used_nodes) > 0:
            PSLG_contour_list.append([unused_nodes[0], list(used_nodes)[0]])
        else:
            # Si aucun node n'est utilisé, créer un edge auto-bouclé (mais ça ne devrait pas arriver)
            PSLG_contour_list.append([unused_nodes[0], unused_nodes[0]])
    else:
        # Tous les nodes sont utilisés, dupliquer un edge existant
        PSLG_contour_list.append(PSLG_contour_list[0].copy())

# Prendre exactement nnod edges
edge_verify = np.array(PSLG_contour_list[:nnod])

# Créer part_verify qui pointe vers les edges du contour dans edge_verify
# part_updated pointe vers les edges dans edge_update, nous devons les mapper vers edge_verify
part_verify = []
for p in part_updated:
    new_part = []
    for edge_idx in p:
        if edge_idx < len(edge_update):
            orig_edge = edge_update[edge_idx]
            # Chercher l'edge correspondant dans edge_verify
            found = False
            for e_idx in range(len(edge_verify)):
                if np.array_equal(edge_verify[e_idx], orig_edge) or np.array_equal(edge_verify[e_idx], orig_edge[::-1]):
                    new_part.append(e_idx)
                    found = True
                    break
            if not found:
                # Si l'edge n'est pas trouvé, c'est qu'il n'est pas dans le contour
                pass
    if len(new_part) > 0:
        # Supprimer les doublons tout en préservant l'ordre
        seen = set()
        unique_new_part = []
        for idx in new_part:
            if idx not in seen:
                seen.add(idx)
                unique_new_part.append(idx)
        part_verify.append(np.array(unique_new_part))
    else:
        part_verify.append(np.array([], dtype=int))

def verify_manifoldness(edge, part):
    nnod = edge.shape[0]

    PSLG_sorted = np.sort(edge, axis=1)
    _, ivec, jvec = np.unique(
        PSLG_sorted, axis=0, return_index=True, return_inverse=True
    )
    edge = edge[ivec, :]

    newpart = []
    for p in part:
        newpart.append(np.unique(jvec[np.array(p)]))
    part = newpart

    # -------------------------------- check part "manifold-ness"
    for p in part:
        eloc = edge[p, :]
        # cast only for bincount computation
        eloc_int = eloc.astype(np.int64, copy=False)
        nadj = np.bincount(eloc_int.ravel(), minlength=nnod)
        if np.any(nadj % 2 != 0):
            print("error")

# Debug
print(f"\nDebug: edge_verify.shape = {edge_verify.shape}, node_update.shape = {node_update.shape}")
print(f"part_verify: {part_verify}")
for i, p in enumerate(part_verify):
    print(f"  Part {i}: {p}")
    if len(p) > 0:
        for j, e_idx in enumerate(p):
            if e_idx < len(edge_verify):
                edge_info = edge_verify[e_idx]
                next_e_idx = p[(j + 1) % len(p)]
                if next_e_idx < len(edge_verify):
                    next_edge_info = edge_verify[next_e_idx]
                    connected = edge_info[1] == next_edge_info[0]
                    print(f"    [{j}] edge[{e_idx}] = {edge_info} -> edge[{next_e_idx}] = {next_edge_info} (connecté: {connected})")

verify_manifoldness(edge_verify, part_verify)
print("SUCCÈS: Le contour est fermé et valide!")
