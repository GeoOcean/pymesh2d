import time

import numpy as np
from scipy.sparse import csr_matrix

from .mesh_cost.triscr import triscr
from .mesh_cost.triarea import triarea
from .mesh_util.deltri import deltri
from .mesh_util.setset import setset
from .mesh_util.tricon import tricon


def smooth(vert=None, conn=None, tria=None, tnum=None, opts=None, hfun=None, harg=[]):
    """
    Perform "hill-climbing" mesh smoothing for two-dimensional 2-simplex triangulations.

    [VERT, EDGE, TRIA, TNUM] = smooth(VERT, EDGE, TRIA, TNUM) returns a
    "smoothed" triangulation {VERT, TRIA}, incorporating optimized vertex
    coordinates and mesh topology.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        XY coordinates of the vertices in the triangulation.
    edge : ndarray of shape (E, 2)
        Array of constrained edges.
    tria : ndarray of shape (T, 3)
        Array of triangles (vertex indices).
    tnum : ndarray of shape (T, 1)
        Array of part indices. Each row of TRIA and EDGE defines an element:
        VERT[TRIA[ii, 0], :], VERT[TRIA[ii, 1], :], and VERT[TRIA[ii, 2], :]
        are the coordinates of the ii-th triangle. The edges in EDGE are
        defined similarly. TNUM[ii] gives the part index of the ii-th triangle.
    opts : dict, optional
        Dictionary containing user-defined parameters:
        - 'vtol' : float, default = 1.0e-2
          Relative vertex movement tolerance. Smoothing converges when
          (VNEW - VERT) <= VTOL * VLEN, where VLEN is a local length scale.
        - 'iter' : int, default = 32
          Maximum number of smoothing iterations.
        - 'disp' : int or float, default = 4
          Display frequency for iteration progress. Set to `np.inf` for quiet execution.
    hfun : callable, optional
        Mesh-size function used for local edge-length control.
    harg : tuple, optional
        Additional arguments passed to the mesh-size function `hfun`.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Updated vertex coordinates after smoothing.
    edge : ndarray of shape (E, 2)
        Updated constrained edges.
    tria : ndarray of shape (T, 3)
        Updated triangle connectivity.
    tnum : ndarray of shape (T, 1)
        Updated part indices.

    Notes
    -----
    - This routine is loosely based on the DISTMESH algorithm,
      employing a spring-based analogy to redistribute mesh vertices.
    - The method introduces a modified spring-based update with
      additional hill-climbing element quality guarantees and
      vertex density control.
    - See: P.-O. Persson and G. Strang (2004),
      "A Simple Mesh Generator in MATLAB", *SIAM Review* 46(2): 329–345.

    References
    ----------
    Translation of the MESH2D function `SMOOTH2` by Darren Engwirda.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ============================================================
    # PHASE 1 : INITIALISATION ET VALIDATION DES ENTRÉES
    # ============================================================
    
    # Initialisation des tableaux vides si les paramètres sont None
    # Cela permet de gérer les cas où certains paramètres ne sont pas fournis
    if vert is None:
        vert = np.empty((0, 2))
    if conn is None:
        conn = np.empty((0, 2), dtype=int)
    if tria is None:
        tria = np.empty((0, 3), dtype=int)
    if tnum is None:
        tnum = np.empty((0, 1), dtype=int)
    if opts is None:
        opts = {}

    # Complétion des options avec les valeurs par défaut si manquantes
    # (vtol, iter, disp, dbug)
    opts = makeopt(opts)

    # ---------------------------------------------- default CONN
    # Si aucune contrainte d'arête n'est fournie, on extrait les arêtes
    # du bord de la triangulation comme contraintes
    if conn.size == 0:
        edge, _ = tricon(tria)  # Obtient toutes les arêtes de la triangulation
        ebnd = edge[:, 3] < 1  # Sélectionne les arêtes de bord (marquées < 1)
        conn = edge[ebnd, 0:2]  # Extrait les indices des sommets de ces arêtes

    # ---------------------------------------------- default TNUM
    # Si aucun numéro de partie n'est fourni, on assigne la partie 1 à tous les triangles
    if tnum.size == 0:
        tnum = np.ones((tria.shape[0], 1), dtype=int)

    # ---------------------------------------------- basic checks
    # Vérification que tous les paramètres sont du bon type (numpy arrays)
    if not (
        isinstance(vert, np.ndarray)
        and isinstance(conn, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(tnum, np.ndarray)
        and isinstance(opts, dict)
    ):
        raise TypeError("smooth:incorrectInputClass - Incorrect input class.")

    nvrt = vert.shape[0]  # Nombre de sommets

    # Vérification que les indices dans conn sont valides (entre 0 et nvrt-1)
    if np.min(conn[:, :2]) < 0 or np.max(conn[:, :2]) > nvrt:
        raise ValueError("smooth:invalidInputs - Invalid EDGE input array.")

    # Vérification que les indices dans tria sont valides
    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) > nvrt:
        raise ValueError("smooth:invalidInputs - Invalid TRIA input array.")

    # ---------------------------------------------- output title
    # Affichage de l'en-tête du tableau de progression si l'affichage est activé
    if not np.isinf(opts["disp"]):
        print("\n Smooth triangulation...\n")
        print(" -------------------------------------------------------")
        print("      |ITER.|          |MOVE(X)|          |DTRI(X)|     ")
        print(" -------------------------------------------------------")

    # ---------------------------------------------- polygon bounds
    # Préparation des données pour la gestion des parties (polygones) du maillage
    # On sauvegarde une copie des sommets et contraintes originaux
    node = vert.copy()  # Sauvegarde des sommets originaux
    PSLG = conn.copy()  # Sauvegarde des contraintes (Planar Straight Line Graph)
    pmax = int(np.max(tnum))  # Nombre maximum de parties dans le maillage
    part = [None for _ in range(pmax)]  # Liste pour stocker les indices de contraintes par partie

    # Pour chaque partie du maillage, on identifie quelles contraintes lui appartiennent
    for ppos in range(pmax):
        tsel = tnum.flatten() == (ppos + 1)  # Triangles appartenant à cette partie
        tcur = tria[tsel, :]  # Triangles de la partie courante
        ecur, tcur = tricon(tcur)  # Arêtes de ces triangles
        ebnd = ecur[:, 3] == -1  # Arêtes de bord (marquées -1)
        same, _ = setset(PSLG, ecur[ebnd, 0:2])  # Trouve les correspondances avec PSLG
        part[ppos] = np.where(same)[0]  # Stocke les indices des contraintes correspondantes

    # ---------------------------------------------- inflate bbox
    # Agrandissement de la boîte englobante pour éviter les problèmes aux bords
    # On calcule les coordonnées min et max du maillage
    vmin = np.min(vert, axis=0)  # Point minimum (coin inférieur gauche)
    vmax = np.max(vert, axis=0)  # Point maximum (coin supérieur droit)

    # On agrandit la boîte de 50% de chaque côté pour créer une zone tampon
    vdel = vmax - 1.0 * vmin  # Dimensions de la boîte
    vmin = vmin - 0.5 * vdel  # Décalage vers le bas-gauche
    vmax = vmax + 0.5 * vdel  # Décalage vers le haut-droite

    # Création d'une boîte rectangulaire avec 4 sommets aux coins
    vbox = np.array(
        [
            [vmin[0], vmin[1]],  # Coin inférieur gauche
            [vmax[0], vmin[1]],  # Coin inférieur droit
            [vmax[0], vmax[1]],  # Coin supérieur droit
            [vmin[0], vmax[1]],  # Coin supérieur gauche
        ]
    )

    # Ajout de ces 4 sommets de la boîte au maillage
    vert = np.vstack((vert, vbox))

    # ============================================================
    # PHASE 2 : BOUCLE PRINCIPALE D'ITÉRATION DE LISSAGE
    # ============================================================
    
    # Initialisation du chronomètre pour mesurer les performances
    tnow = time.time()
    tcpu = {
        "full": 0.0,  # Temps total
        "dtri": 0.0,  # Temps pour la triangulation de Delaunay
        "tcon": 0.0,  # Temps pour la construction de la connectivité
        "iter": 0.0,  # Temps pour les itérations de lissage
        "undo": 0.0,  # Temps pour l'annulation (hill-climbing)
        "keep": 0.0,  # Temps pour le contrôle de densité
    }

    # Boucle principale : itère jusqu'à convergence ou nombre max d'itérations
    for iter in range(int(opts["iter"])):
        # ------------------------------------------ inflate adj.
        # Reconstruction de la connectivité arêtes-triangles à chaque itération
        # car le maillage peut changer (sommets ajoutés/supprimés)
        ttic = time.time()
        edge, tria = tricon(tria, conn)  # Obtient toutes les arêtes et met à jour tria
        tcpu["tcon"] += time.time() - ttic

        # ------------------------------------------ compute scr.
        # Calcul du score de qualité des triangles AVANT le lissage
        # Le score mesure la qualité géométrique (forme) de chaque triangle
        oscr = triscr(vert, tria)

        # ------------------------------------------ vert. iter's
        # ITÉRATIONS LOCALES : Lissage basé sur l'analogie des ressorts
        # Cette partie déplace les sommets pour améliorer la régularité du maillage
        ttic = time.time()
        nvrt = vert.shape[0]  # Nombre de sommets
        nedg = edge.shape[0]  # Nombre d'arêtes

        # Construction de matrices creuses pour représenter la connectivité sommet-arête
        # IMAT : matrice indiquant quels sommets sont au début de chaque arête
        IMAT = csr_matrix(
            (np.ones(nedg), (edge[:, 0], np.arange(nedg))), shape=(nvrt, nedg)
        )
        # JMAT : matrice indiquant quels sommets sont à la fin de chaque arête
        JMAT = csr_matrix(
            (np.ones(nedg), (edge[:, 1], np.arange(nedg))), shape=(nvrt, nedg)
        )

        # EMAT : matrice d'incidence sommet-arête (combinaison de IMAT et JMAT)
        # Chaque ligne correspond à un sommet, chaque colonne à une arête
        EMAT = IMAT + JMAT
        vdeg = np.array(EMAT.sum(axis=1)).flatten()  # Degré de chaque sommet (nb d'arêtes connectées)
        free = vdeg == 0  # Sommets isolés (non connectés)

        vold = vert.copy()  # Sauvegarde des positions avant modification

        # Local iterations
        # Sous-itérations pour affiner le lissage (jusqu'à 8 itérations selon le numéro d'itération)
        for isub in range(max(1, min(8, iter))):
            # compute HFUN at vert/midpoints
            # Évaluation de la fonction de taille de maille (hfun) aux sommets
            # Cette fonction contrôle la taille désirée des arêtes localement
            hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
            # Taille de maille au milieu de chaque arête (moyenne des deux extrémités)
            hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])
            
            # calc. relative edge extensions
            # Calcul des vecteurs et longueurs des arêtes
            evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]  # Vecteur de chaque arête
            elen = np.sqrt(np.sum(evec**2, axis=1))  # Longueur de chaque arête

            # Calcul du facteur d'extension relative : mesure si l'arête est trop courte/longue
            # scal = 1.0 signifie longueur parfaite, > 0 si trop courte, < 0 si trop longue
            scal = 1.0 - elen / hmid
            scal = np.clip(scal, -1.0, 1.0)  # Limite entre -1 et 1
            
            # projected points from each end
            # Calcul de nouvelles positions projetées pour chaque extrémité d'arête
            # Le facteur 0.67 contrôle l'amplitude du déplacement
            ipos = vert[edge[:, 0], :] - 0.67 * (scal[:, None] * evec)  # Position depuis le début
            jpos = vert[edge[:, 1], :] + 0.67 * (scal[:, None] * evec)  # Position depuis la fin
            
            # scal = ...                      nlin. weight
            # Transformation du facteur en poids non-linéaire pour le calcul
            # On utilise la valeur absolue élevée à la puissance 1 (linéaire)
            # Le epsilon évite la division par zéro
            scal = np.maximum(np.abs(scal) ** 1, np.finfo(float).eps ** 0.75)
            
            # sum contributions edge-to-vert
            # Agrégation de toutes les contributions des arêtes vers chaque sommet
            # Chaque sommet reçoit une contribution pondérée de toutes ses arêtes connectées
            vnew = IMAT.dot(scal[:, None] * ipos) + JMAT.dot(scal[:, None] * jpos)
            vsum = np.maximum(EMAT.dot(scal), np.finfo(float).eps ** 0.75)  # Somme des poids

            # Normalisation : moyenne pondérée des positions proposées
            vnew = vnew / vsum[:, None]
            
            # fixed points. edge projection?
            # Les sommets contraints (sur les bords) ne doivent pas bouger
            vnew[conn.flatten(), :] = vert[conn.flatten(), :]
            # Les sommets isolés ne bougent pas non plus
            vnew[vdeg == 0, :] = vert[vdeg == 0, :]
            
            # reset for the next local iter.
            # Mise à jour des positions pour la prochaine sous-itération
            vert = vnew

        tcpu["iter"] += time.time() - ttic
        
        # ------------------------------------------ hill-climber
        # ALGORITHME HILL-CLIMBING : Annulation des modifications qui dégradent la qualité
        # Cette partie vérifie que le lissage n'a pas dégradé la qualité des triangles
        ttic = time.time()

        # unwind vert. upadte if score lower
        # Initialisation des scores et du masque des triangles à vérifier
        nscr = np.ones(tria.shape[0])  # Nouveaux scores (initialisés à 1)
        btri = np.ones(tria.shape[0], dtype=bool)  # Masque : tous les triangles au début

        umax = 8  # Nombre maximum de tentatives d'annulation
        for undo in range(umax):
            # Calcul du score de qualité pour les triangles marqués
            nscr[btri] = triscr(vert, tria[btri, :])

            # TRUE if tria needs "unwinding"
            # Définition d'un seuil de tolérance qui augmente avec les itérations
            # Au début, on est strict (0.70), puis on devient plus tolérant (jusqu'à 0.90)
            smin = 0.70  # Seuil minimum
            smax = 0.90  # Seuil maximum
            sdel = 0.025  # Incrément par itération

            stol = smin + iter * sdel  # Seuil adaptatif
            stol = min(smax, stol)  # Ne dépasse pas le maximum

            # Identification des triangles dont la qualité a baissé et est sous le seuil
            btri = (nscr <= stol) & (nscr < oscr)

            # Si aucun triangle ne nécessite d'annulation, on arrête
            if not np.any(btri):
                break

            # relax toward old vert. coord's
            # Identification des sommets des triangles à annuler
            ivrt = np.unique(tria[btri, :3])  # Sommets uniques de ces triangles
            bvrt = np.zeros(vert.shape[0], dtype=bool)
            bvrt[ivrt] = True  # Masque des sommets à corriger

            # Calcul des poids pour l'interpolation entre ancienne et nouvelle position
            # Plus on itère, plus on revient vers l'ancienne position (bold augmente)
            if undo != umax:
                bnew = 0.75**undo  # Poids de la nouvelle position (décroît)
                bold = 1.0 - bnew  # Poids de l'ancienne position
            else:
                bnew = 0.0  # Dernière tentative : on revient complètement à l'ancienne
                bold = 1.0 - bnew

            # Interpolation linéaire : retour partiel vers l'ancienne position
            vert[bvrt, :] = bold * vold[bvrt, :] + bnew * vert[bvrt, :]

            # Mise à jour du masque : inclure tous les triangles touchant ces sommets
            btri = np.any(bvrt[tria[:, :3]], axis=1)

        oscr = nscr  # Mise à jour des scores de référence
        tcpu["undo"] += time.time() - ttic

        # ------------------------------------- test convergence!
        # CONTRÔLE DE CONVERGENCE ET DE DENSITÉ
        # Cette partie mesure le mouvement des sommets et contrôle la densité du maillage
        ttic = time.time()

        # Calcul du déplacement carré de chaque sommet depuis la dernière itération
        vdel = np.sum((vert - vold) ** 2, axis=1)

        # Recalcul des longueurs d'arêtes pour le contrôle de densité
        evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
        elen = np.sqrt(np.sum(evec**2, axis=1))

        # Recalcul de la fonction de taille de maille
        hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
        hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])
        # Ratio longueur réelle / longueur désirée pour chaque arête
        scal = elen / hmid

        # Calcul des points milieux de chaque arête (pour fusion éventuelle)
        emid = 0.5 * (vert[edge[:, 0], :] + vert[edge[:, 1], :])

        # ------------------------------------- |deg|-based prune
        # DÉCISION : Quels sommets garder dans le maillage ?
        # On garde les sommets avec un degré élevé (> 4), les sommets contraints, et les isolés
        keep = np.zeros(vert.shape[0], dtype=bool)
        keep[vdeg > 4] = True  # Sommets avec beaucoup de connexions (importants)
        keep[conn.flatten()] = True  # Sommets contraints (ne peuvent pas être supprimés)
        keep[free.flatten()] = True  # Sommets isolés (à garder aussi)

        # ------------------------------------- 'density' control
        # CONTRÔLE DE DENSITÉ : Identification des arêtes trop courtes ou trop longues
        # Les arêtes trop courtes peuvent être fusionnées, les trop longues divisées
        lmax = 5.0 / 4.0  # Seuil supérieur : arête > 1.25 * longueur désirée = trop longue
        lmin = 1.0 / lmax  # Seuil inférieur : arête < 0.8 * longueur désirée = trop courte

        less = scal <= lmin  # Arêtes trop courtes (à fusionner potentiellement)
        more = scal >= lmax  # Arêtes trop longues (à diviser potentiellement)

        # Exclusion des arêtes sur les bords du domaine (ne pas modifier)
        vbnd = np.zeros(vert.shape[0], dtype=bool)
        vbnd[conn[:, 0]] = True  # Marque les sommets de bord
        vbnd[conn[:, 1]] = True

        ebad = vbnd[edge[:, 0]] | vbnd[edge[:, 1]]  # Arêtes touchant le bord

        less[ebad] = False  # Ne pas fusionner les arêtes de bord
        more[ebad] = False  # Ne pas diviser les arêtes de bord

        # ------------------------------------- force as disjoint
        # ASSURANCE QUE LES ARÊTES À FUSIONNER SONT DISJOINTES
        # On ne peut fusionner deux arêtes que si leurs deux sommets peuvent être supprimés
        lidx = np.where(less)[0]  # Indices des arêtes à fusionner
        for epos in lidx:
            inod = edge[epos, 0]  # Premier sommet de l'arête
            jnod = edge[epos, 1]  # Second sommet de l'arête
            # --------------------------------- if still disjoint
            # Si les deux sommets sont marqués pour être gardés, on ne peut pas fusionner
            # On les retire donc de la liste "keep" pour permettre la fusion
            if keep[inod] and keep[jnod]:
                keep[inod] = False
                keep[jnod] = False
            else:
                # Si l'un des deux doit être gardé, on ne peut pas fusionner cette arête
                less[epos] = False

        # Éviter les conflits : si une arête "less" a ses sommets gardés, annuler "more"
        ebad = keep[edge[less, 0]] & keep[edge[less, 1]]
        indices = np.flatnonzero(ebad)
        indices = indices[indices < more.size]
        more[indices] = False

        # ------------------------------------- reindex vert/tria
        # PREMIÈRE TENTATIVE DE FUSION : Création d'un nouveau maillage avec fusion
        # On crée un nouveau maillage où les arêtes courtes sont fusionnées en leur milieu
        redo = np.zeros(vert.shape[0], dtype=int)  # Table de réindexation
        itop = np.count_nonzero(keep)  # Nombre de sommets à garder
        iend = np.count_nonzero(less)  # Nombre d'arêtes à fusionner

        # Construction de la table de réindexation
        redo[keep] = np.arange(itop)  # Les sommets gardés gardent leurs indices initiaux
        # Les deux sommets d'une arête fusionnée pointent vers le même nouvel indice (milieu)
        redo[edge[less, 0]] = np.arange(itop, itop + iend)  # Vers nouveaux milieux
        redo[edge[less, 1]] = np.arange(itop, itop + iend)

        # Création du nouveau maillage : sommets gardés + milieux des arêtes fusionnées
        vnew = np.vstack([vert[keep, :], emid[less, :]])
        # Réindexation des triangles avec la nouvelle numérotation
        tnew = redo[tria[:, 0:3]]

        # Filtrage des triangles dégénérés (collapsed)
        # Un triangle est dégénéré si deux sommets sont identiques
        ttmp = np.sort(tnew, axis=1)  # Tri pour faciliter la détection
        okay = np.all(np.diff(ttmp, axis=1) != 0, axis=1)  # Vérifie que tous les sommets sont différents
        okay = okay & (ttmp[:, 0] > 0)  # Évite les triangles avec indices négatifs
        tnew = tnew[okay, :]  # Garde seulement les triangles valides

        # ------------------------------------- quality preserver
        # VÉRIFICATION DE QUALITÉ : Le nouveau maillage est-il meilleur ?
        # On calcule les scores de qualité du nouveau maillage
        nscr = triscr(vnew, tnew)

        stol = 0.80  # Seuil de qualité minimum acceptable
        # Triangles dont la qualité a baissé par rapport à l'ancien maillage
        tbad = (nscr < stol) & (nscr < oscr[okay])

        # Identification des sommets des triangles de mauvaise qualité
        vbad = np.zeros(vnew.shape[0], dtype=bool)
        vbad[tnew[tbad, :]] = True

        # ------------------------------------- filter edge merge
        # FILTRAGE : Annulation des fusions qui dégradent la qualité
        # Si la fusion d'une arête crée des triangles de mauvaise qualité, on l'annule
        lidx = np.where(less)[0]  # Indices des arêtes fusionnées
        # Vérifie si les nouveaux sommets (milieux) créent de mauvais triangles
        ebad = vbad[redo[edge[lidx, 0]]] | vbad[redo[edge[lidx, 1]]]

        # Annulation des fusions problématiques
        less[lidx[ebad]] = False
        # Remise des sommets originaux dans la liste "keep"
        keep[edge[lidx[ebad], 0:2].flatten()] = True

        # ------------------------------------- reindex vert/conn
        # RÉINDEXATION FINALE : Construction du maillage final avec fusions et divisions
        redo = np.zeros(vert.shape[0], dtype=int)
        itop = np.count_nonzero(keep)  # Sommets à garder
        iend = np.count_nonzero(less)  # Arêtes à fusionner

        # Table de réindexation finale
        redo[keep] = np.arange(itop)
        redo[edge[less, 0]] = np.arange(itop, itop + iend)  # Fusion vers milieux
        redo[edge[less, 1]] = np.arange(itop, itop + iend)

        # Maillage final : sommets gardés + milieux des fusions + milieux des divisions
        vert = np.vstack([vert[keep, :], emid[less, :], emid[more, :]])
        # Réindexation des contraintes
        conn = redo[conn]

        tcpu["keep"] += time.time() - ttic

        # ------------------------------------- build current CDT
        # RECONSTRUCTION DE LA TRIANGULATION DE DELAUNAY
        # Après avoir modifié les sommets (fusion/division), on doit reconstruire la triangulation
        # La triangulation de Delaunay garantit une bonne qualité géométrique
        ttic = time.time()
        vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
        tcpu["dtri"] += time.time() - ttic

        # ------------------------------------- dump-out progress
        # AFFICHAGE DE LA PROGRESSION ET TEST DE CONVERGENCE
        # Normalisation du déplacement par la taille locale de maille
        vdel = vdel / (hvrt.flatten() ** 2)
        # Un sommet a bougé si son déplacement normalisé dépasse la tolérance
        move = vdel > opts["vtol"] ** 2

        nmov = np.count_nonzero(move)  # Nombre de sommets qui ont encore bougé
        ntri = tria.shape[0]  # Nombre de triangles

        # Affichage périodique de la progression
        if iter % opts["disp"] == 0:
            print(f"{iter:11d} {nmov:18d} {ntri:18d}")

        # ------------------------------------- loop convergence!
        # Si aucun sommet n'a bougé, on a convergé : arrêt de la boucle
        if nmov == 0:
            break

    # ============================================================
    # PHASE 3 : NETTOYAGE FINAL
    # ============================================================
    
    # Extraction des 3 premiers indices de chaque triangle (format standard)
    tria = tria[:, 0:3]

    # ----------------------------------------- prune unused vert
    # SUPPRESSION DES SOMMETS INUTILISÉS
    # Après toutes les modifications, certains sommets peuvent ne plus être utilisés
    # (par exemple, les sommets de la boîte ajoutée au début)
    keep = np.zeros(vert.shape[0], dtype=bool)
    keep[tria.flatten()] = True  # Garde les sommets utilisés dans les triangles
    keep[conn.flatten()] = True  # Garde les sommets des contraintes

    # Réindexation finale pour compacter les indices
    redo = np.zeros(vert.shape[0], dtype=int)
    redo[keep] = np.arange(np.count_nonzero(keep))

    # Mise à jour des indices dans conn et tria
    conn = redo[conn]
    tria = redo[tria]

    # Suppression des sommets inutilisés
    vert = vert[keep, :]

    # ============================================================
    # PHASE 4 : CORRECTION DES SMALL FLOW LINKS
    # ============================================================
    # Vérification et correction des small flow links jusqu'à ce qu'il n'y en ait plus
    # Approche simplifiée : supprimer les triangles problématiques ou déplacer les sommets libres
    try:
        from .mesh_util.circo import small_flow_links, circumcenters
        
        max_fix_iter = 100  # Nombre maximum d'itérations de correction
        removesmalllinkstrsh = opts.get("removesmalllinkstrsh", 0.1)
        
        fix_iter = 0
        stagnation_count = 0
        
        for fix_iter in range(max_fix_iter):
            # Reconstruire la connectivité pour vérifier
            edge_fix, tria_6col_fix = tricon(tria, conn)
            
            # Vérifier s'il y a des small flow links
            nlinktoosmall, small_link_indices = small_flow_links(
                vert, tria, edge_fix, removesmalllinkstrsh, conn
            )
            
            if nlinktoosmall == 0:
                if not np.isinf(opts["disp"]) and fix_iter > 0:
                    print(f"\n All small flow links fixed after {fix_iter} iterations")
                break
            
            # Afficher la progression
            if not np.isinf(opts["disp"]) and fix_iter % 10 == 0:
                print(f" Fix iteration {fix_iter}: {nlinktoosmall} small flow links")
            
            # Si on stagne trop, supprimer directement les triangles problématiques
            if stagnation_count >= 3 or (nlinktoosmall <= 3 and fix_iter > 5):
                # Mode suppression directe
                if not np.isinf(opts["disp"]):
                    print(f"     [MODE SUPPRESSION] stagnation={stagnation_count}, links={nlinktoosmall}")
                problem_edges = edge_fix[small_link_indices]
                t1_indices = problem_edges[:, 2].astype(int)
                t2_indices = problem_edges[:, 3].astype(int)
                
                problem_triangles = set()
                valid_mask = (
                    (t1_indices >= 0) & (t1_indices < len(tria)) &
                    (t2_indices >= 0) & (t2_indices < len(tria))
                )
                
                tri_areas_simple = np.abs(triarea(vert, tria))
                
                for idx in np.where(valid_mask)[0]:
                    t1_idx = t1_indices[idx]
                    t2_idx = t2_indices[idx]
                    # Supprimer le plus petit triangle
                    if tri_areas_simple[t1_idx] < tri_areas_simple[t2_idx]:
                        problem_triangles.add(t1_idx)
                    else:
                        problem_triangles.add(t2_idx)
                
                if len(problem_triangles) > 0:
                    if not np.isinf(opts["disp"]):
                        print(f"     [MODE SUPPRESSION] Removing {len(problem_triangles)} triangle(s)")
                    problem_triangles = np.array(list(problem_triangles))
                    keep_triangles = np.ones(len(tria), dtype=bool)
                    keep_triangles[problem_triangles] = False
                    tria = tria[keep_triangles, :]
                    if len(tnum) == len(keep_triangles):
                        tnum = tnum[keep_triangles]
                    vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
                    tria = tria[:, 0:3]
                    stagnation_count = 0
                    continue
            
            # Mode déplacement : calculer les circumcenters et déplacer les sommets libres
            if not np.isinf(opts["disp"]):
                print(f"     [MODE DEPLACEMENT] Processing {nlinktoosmall} problematic link(s)")
            with np.errstate(invalid='ignore', divide='ignore'):
                try:
                    bb_fix = circumcenters(vert, tria, edge_fix, tria_6col_fix)
                    circumcenters_pos = bb_fix[:, 0:2]
                    # Filtrer les valeurs invalides
                    valid_cc = np.isfinite(circumcenters_pos).all(axis=1)
                    if not np.all(valid_cc):
                        # Si certains circumcenters sont invalides, utiliser le mode suppression
                        if not np.isinf(opts["disp"]):
                            print(f"     [MODE DEPLACEMENT] Invalid circumcenters, skipping")
                        stagnation_count += 1
                        continue
                except Exception as e:
                    if not np.isinf(opts["disp"]):
                        print(f"     [MODE DEPLACEMENT] Exception in circumcenters: {e}")
                    stagnation_count += 1
                    continue
            
            tri_areas_fix = np.abs(triarea(vert, tria))
            conn_flat = set(conn.flatten())
            
            # Dictionnaire simplifié pour les déplacements
            vertex_displacements = {}
            vertex_counts = {}
            
            # Triangles entièrement contraints nécessitant insertion de nœud
            constrained_triangles_to_split = set()
            
            # Compteurs pour statistiques
            stats_all_constrained = 0
            stats_one_free = 0
            stats_both_free = 0
            stats_skipped = 0
            
            problem_edges = edge_fix[small_link_indices]
            t1_indices = problem_edges[:, 2].astype(int)
            t2_indices = problem_edges[:, 3].astype(int)
            edge_vertices = problem_edges[:, 0:2].astype(int)
            
            valid_mask = (
                (t1_indices >= 0) & (t1_indices < len(tria)) &
                (t2_indices >= 0) & (t2_indices < len(tria))
            )
            
            n_valid_links = np.sum(valid_mask)
            if not np.isinf(opts["disp"]):
                print(f"     [MODE DEPLACEMENT] Valid links to process: {n_valid_links}")
            
            for idx in np.where(valid_mask)[0]:
                t1_idx = t1_indices[idx]
                t2_idx = t2_indices[idx]
                shared_v1 = edge_vertices[idx, 0]
                shared_v2 = edge_vertices[idx, 1]
                
                # Vérifications de sécurité
                if (t1_idx >= len(tria) or t2_idx >= len(tria) or
                    shared_v1 >= len(vert) or shared_v2 >= len(vert) or
                    t1_idx < 0 or t2_idx < 0 or shared_v1 < 0 or shared_v2 < 0):
                    stats_skipped += 1
                    continue
                
                # Obtenir les sommets
                tri1_verts = tria[t1_idx, :]
                tri2_verts = tria[t2_idx, :]
                
                # Identifier les sommets opposés
                opp1 = None
                for v in tri1_verts:
                    if v != shared_v1 and v != shared_v2:
                        opp1 = v
                        break
                
                opp2 = None
                for v in tri2_verts:
                    if v != shared_v1 and v != shared_v2:
                        opp2 = v
                        break
                
                if (opp1 is None or opp2 is None or 
                    opp1 >= len(vert) or opp2 >= len(vert) or
                    opp1 < 0 or opp2 < 0):
                    stats_skipped += 1
                    continue
                
                # Vérifier les circumcenters
                if (t1_idx >= len(circumcenters_pos) or t2_idx >= len(circumcenters_pos) or
                    not np.isfinite(circumcenters_pos[t1_idx]).all() or
                    not np.isfinite(circumcenters_pos[t2_idx]).all()):
                    stats_skipped += 1
                    continue
                
                cc1 = circumcenters_pos[t1_idx]
                cc2 = circumcenters_pos[t2_idx]
                
                # Distance actuelle avec vérification
                cc_diff = cc2 - cc1
                link_distance = np.linalg.norm(cc_diff)
                if not np.isfinite(link_distance) or link_distance < 1e-12:
                    link_distance = 1e-12
                
                # Distance minimale désirée avec vérification
                area1 = max(tri_areas_fix[t1_idx], 1e-12)
                area2 = max(tri_areas_fix[t2_idx], 1e-12)
                if not (np.isfinite(area1) and np.isfinite(area2) and area1 > 0 and area2 > 0):
                    stats_skipped += 1
                    continue
                
                sqrt_ba = np.sqrt(area1) + np.sqrt(area2)
                if sqrt_ba < 1e-12:
                    stats_skipped += 1
                    continue
                
                dxlim = 0.9 * removesmalllinkstrsh * 0.5 * sqrt_ba
                
                if link_distance >= dxlim or dxlim < 1e-12:
                    stats_skipped += 1
                    continue
                
                # Distance nécessaire - très agressif
                needed_distance = (dxlim - link_distance) * 5.0
                
                # Direction de séparation avec vérification
                cc_dir_norm = link_distance
                if cc_dir_norm < 1e-12:
                    # Circumcenters confondus : direction perpendiculaire à l'arête
                    edge_vec = vert[shared_v2] - vert[shared_v1]
                    edge_len = np.linalg.norm(edge_vec)
                    if edge_len < 1e-12:
                        stats_skipped += 1
                        continue
                    cc_dir = np.array([-edge_vec[1], edge_vec[0]]) / edge_len
                else:
                    cc_dir = cc_diff / cc_dir_norm
                
                # Identifier les sommets libres
                opp1_free = opp1 not in conn_flat
                opp2_free = opp2 not in conn_flat
                
                if not opp1_free and not opp2_free:
                    # Tous contraints : marquer le triangle le plus petit pour insertion de nœud
                    stats_all_constrained += 1
                    if area1 < area2:
                        constrained_triangles_to_split.add(t1_idx)
                    else:
                        constrained_triangles_to_split.add(t2_idx)
                    continue
                
                # Déplacement très agressif
                displacement_magnitude = needed_distance * 1.0
                if opp1_free and not opp2_free:
                    # Seul opp1 libre : déplacement très agressif
                    stats_one_free += 1
                    displacement_magnitude = needed_distance * 10.0
                elif opp2_free and not opp1_free:
                    # Seul opp2 libre : déplacement très agressif
                    stats_one_free += 1
                    displacement_magnitude = needed_distance * 10.0
                else:
                    # Les deux libres
                    stats_both_free += 1
                
                # Accumuler les déplacements
                if opp1_free:
                    if opp1 not in vertex_displacements:
                        vertex_displacements[opp1] = np.zeros(2)
                        vertex_counts[opp1] = 0
                    vertex_displacements[opp1] -= cc_dir * displacement_magnitude * 0.5
                    vertex_counts[opp1] += 1
                
                if opp2_free:
                    if opp2 not in vertex_displacements:
                        vertex_displacements[opp2] = np.zeros(2)
                        vertex_counts[opp2] = 0
                    vertex_displacements[opp2] += cc_dir * displacement_magnitude * 0.5
                    vertex_counts[opp2] += 1
            
            # Afficher les statistiques
            if not np.isinf(opts["disp"]) and n_valid_links > 0:
                print(f"     [STATS] all_constrained={stats_all_constrained}, one_free={stats_one_free}, both_free={stats_both_free}, skipped={stats_skipped}")
            
            # Insérer des nœuds au centre des triangles entièrement contraints
            if len(constrained_triangles_to_split) > 0:
                if not np.isinf(opts["disp"]):
                    print(f"     [INSERTION] {len(constrained_triangles_to_split)} constrained triangle(s) to split")
                new_vertices = []
                for tri_idx in constrained_triangles_to_split:
                    if tri_idx >= len(tria) or tri_idx < 0:
                        continue
                    tri_verts = tria[tri_idx, :]
                    if np.any(tri_verts >= len(vert)) or np.any(tri_verts < 0):
                        continue
                    # Calculer le barycentre (centre du triangle)
                    v0 = vert[tri_verts[0], :]
                    v1 = vert[tri_verts[1], :]
                    v2 = vert[tri_verts[2], :]
                    if np.all(np.isfinite(v0)) and np.all(np.isfinite(v1)) and np.all(np.isfinite(v2)):
                        center = (v0 + v1 + v2) / 3.0
                        if np.all(np.isfinite(center)):
                            new_vertices.append(center)
                
                if len(new_vertices) > 0:
                    new_vertices = np.array(new_vertices)
                    # Ajouter les nouveaux nœuds (non contraints)
                    vert = np.vstack([vert, new_vertices])
                    if not np.isinf(opts["disp"]):
                        print(f"     [INSERTION] Inserted {len(new_vertices)} node(s) in constrained triangles")
                    # Reconstruire la triangulation
                    vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
                    tria = tria[:, 0:3]
                    stagnation_count = 0
                    continue
            
            # Appliquer les déplacements
            any_moved = False
            n_vertices_moved = 0
            for v_idx, disp in vertex_displacements.items():
                if v_idx not in conn_flat and 0 <= v_idx < len(vert):
                    count = max(vertex_counts.get(v_idx, 1), 1)
                    normalized_disp = disp / count
                    # Facteur très agressif
                    relaxation = min(2.0 + 0.1 * fix_iter, 10.0)
                    vert[v_idx] += normalized_disp * relaxation
                    any_moved = True
                    n_vertices_moved += 1
            
            if any_moved:
                if not np.isinf(opts["disp"]):
                    print(f"     [DEPLACEMENT] Moved {n_vertices_moved} vertex/vertices")
                stagnation_count = 0
                # Reconstruire la triangulation
                vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
                tria = tria[:, 0:3]
            else:
                if not np.isinf(opts["disp"]):
                    print(f"     [DEPLACEMENT] No vertices moved, stagnation_count={stagnation_count+1}")
                stagnation_count += 1
                if stagnation_count >= 2:
                    # Si on stagne, passer au mode suppression
                    if not np.isinf(opts["disp"]):
                        print(f"     [DEPLACEMENT] Stagnation detected, will switch to deletion mode")
                    continue
        
        # Vérification finale
        edge_final, _ = tricon(tria, conn)
        nlinktoosmall_final, _ = small_flow_links(
            vert, tria, edge_final, removesmalllinkstrsh, conn
        )
        if nlinktoosmall_final > 0 and not np.isinf(opts["disp"]):
            print(f"\n Warning: {nlinktoosmall_final} small flow links remain after {max_fix_iter} correction iterations")
        elif nlinktoosmall_final == 0 and not np.isinf(opts["disp"]) and fix_iter > 0:
            print(f"\n All small flow links fixed after {fix_iter} iterations")
    except (ImportError, Exception) as e:
        if opts.get("dbug", False):
            print(f"\n Small flow links correction failed: {e}")
        pass

    # Calcul du temps total
    tcpu["full"] += time.time() - tnow

    # Affichage des statistiques de performance si demandé
    if opts["dbug"]:
        print("\n Mesh smoothing timer...\n")
        print(f" FULL: {tcpu['full']:.6f}")
        print(f" DTRI: {tcpu['dtri']:.6f}")
        print(f" TCON: {tcpu['tcon']:.6f}")
        print(f" ITER: {tcpu['iter']:.6f}")
        print(f" UNDO: {tcpu['undo']:.6f}")
        print(f" KEEP: {tcpu['keep']:.6f}\n")

    if not np.isinf(opts["disp"]):
        print("")

    return vert, conn, tria, tnum


def evalhfn(vert, edge, EMAT, hfun=None, harg=[]):
    """
    Evaluate the mesh spacing function (spacing-function) at mesh vertices.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        XY coordinates of the mesh vertices.
    edge : ndarray of shape (E, 2)
        Array of edge connections.
    EMAT : ndarray or scipy.sparse matrix
        Vertex–edge incidence matrix.
    hfun : float, callable, or None
        Mesh-size function or constant spacing value.
    harg : tuple
        Additional arguments passed to the mesh-size function `hfun`.

    Returns
    -------
    hvrt : ndarray of shape (N,)
        Mesh-size function values evaluated at the vertices.

    References
    ----------
    Translation of the MESH2D function `EVALHFN` by Darren Engwirda.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if hfun is not None and (np.isscalar(hfun) or callable(hfun)):
        if np.isscalar(hfun):
            hvrt = hfun * np.ones(vert.shape[0])
        else:
            hvrt = hfun(vert, *harg)
    else:
        # no HFUN - HVRT is mean edge-len. at vertices!
        evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
        elen = np.sqrt(np.sum(evec**2, axis=1))

        hvrt = np.ravel(EMAT @ elen) / np.maximum(
            np.ravel(np.sum(EMAT, axis=1)), np.finfo(float).eps
        )

        free = np.ones(vert.shape[0], dtype=bool)
        free[edge[:, 0]] = False
        free[edge[:, 1]] = False

        hvrt[free] = np.inf

    return hvrt


def makeopt(opts=None):
    """
    Initialize the options structure for the `smooth` function.

    Parameters
    ----------
    opts : dict or None
        User-defined options dictionary. If None, a new dictionary is created.

    Returns
    -------
    opts : dict
        Options dictionary completed with default values for missing parameters.

    References
    ----------
    Translation of the MESH2D function `MAKEOPT` by Darren Engwirda.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if opts is None:
        opts = {}

    # --------------------------- ITER
    if "iter" not in opts:
        opts["iter"] = 32
    else:
        if not isinstance(opts["iter"], (int, float)):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if not (
            isinstance(opts["iter"], (int, float))
            and not isinstance(opts["iter"], bool)
        ):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["iter"], (list, tuple)) or hasattr(opts["iter"], "__len__"):
            raise ValueError("smooth:incorrectDimensions - Incorrect input dimensions.")
        if opts["iter"] <= 0:
            raise ValueError("smooth:invalidOptionValues - Invalid OPT.ITER selection.")

    # --------------------------- DISP
    if "disp" not in opts:
        opts["disp"] = 4
    else:
        if not isinstance(opts["disp"], (int, float)):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["disp"], (list, tuple)) or hasattr(opts["disp"], "__len__"):
            raise ValueError("smooth:incorrectDimensions - Incorrect input dimensions.")
        if opts["disp"] <= 0:
            raise ValueError("smooth:invalidOptionValues - Invalid OPT.DISP selection.")

    # --------------------------- VTOL
    if "vtol" not in opts:
        opts["vtol"] = 1.0e-2
    else:
        if not isinstance(opts["vtol"], (int, float)):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["vtol"], (list, tuple)) or hasattr(opts["vtol"], "__len__"):
            raise ValueError("smooth:incorrectDimensions - Incorrect input dimensions.")
        if opts["vtol"] <= 0:
            raise ValueError("smooth:invalidOptionValues - Invalid OPT.VTOL selection.")

    # --------------------------- DBUG
    if "dbug" not in opts:
        opts["dbug"] = False
    else:
        if not isinstance(opts["dbug"], bool):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["dbug"], (list, tuple)) or hasattr(opts["dbug"], "__len__"):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")

    return opts
