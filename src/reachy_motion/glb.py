"""Export the Reachy Mini as a GLB with a head/body/antenna node hierarchy.

The GLB is driven directly in the browser (Three.js) from a move's per-frame data:
``body`` rotates by ``body_yaw``, ``head`` gets the 4x4 head pose, ``antenna_l/r`` rotate
by their angle. No IK and no physics — so it's always stable (unlike driving the Stewart
platform), at the cost of not animating the internal leg linkage (which we hide).

Node hierarchy (names matter — Three.js looks them up):
    base                      static foot/base shell
    body                      rotates about Z (body_yaw), pivot at home body frame
      head                    gets the head 4x4 (relative to head frame at z=0.1496)
        antenna_l, antenna_r  rotate about their joint axis
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .viewer import model_path

# MuJoCo bodies grouped into logical, animatable nodes. Everything else (the 6
# Stewart legs + horns) is hidden — it would need IK to animate and isn't visible.
GROUPS = {
    "base": ["body_foot_3dprint"],
    "body": ["body_down_3dprint"],
    "head": ["xl_330"],
    "antenna_r": ["dc15_a01_horn_dummy_7"],
    "antenna_l": ["dc15_a01_horn_dummy_8"],
}
PARENT = {"base": None, "body": None, "head": "body", "antenna_l": "head", "antenna_r": "head"}

GLB_PATH = Path(__file__).resolve().parent.parent.parent / "out" / "reachy_mini.glb"


def _mat(pos, xmat) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(xmat).reshape(3, 3)
    T[:3, 3] = pos
    return T


def _geom_mesh(m, d, gi: int) -> trimesh.Trimesh | None:
    """Build a world-space trimesh for a mesh geom at the current (home) pose."""
    import mujoco

    if m.geom_type[gi] != mujoco.mjtGeom.mjGEOM_MESH:
        return None
    did = m.geom_dataid[gi]
    if did < 0:
        return None
    v0 = m.mesh_vertadr[did]
    nv = m.mesh_vertnum[did]
    f0 = m.mesh_faceadr[did]
    nf = m.mesh_facenum[did]
    verts = m.mesh_vert[v0 : v0 + nv].reshape(-1, 3).astype(np.float64)
    faces = m.mesh_face[f0 : f0 + nf].reshape(-1, 3)
    # geom world transform at current pose
    gpos = d.geom_xpos[gi]
    gmat = d.geom_xmat[gi].reshape(3, 3)
    world = (gmat @ verts.T).T + gpos
    mesh = trimesh.Trimesh(vertices=world, faces=faces, process=False)
    # Real colors live in materials (geom_rgba is the default gray); prefer mat_rgba.
    matid = int(m.geom_matid[gi])
    rgba = m.mat_rgba[matid] if matid >= 0 else m.geom_rgba[gi]
    mesh.visual.vertex_colors = np.tile((rgba * 255).astype(np.uint8), (len(world), 1))
    return mesh


def export_glb(out_path: Path | str = GLB_PATH) -> Path:
    """Build and write the Reachy Mini GLB with the animatable node hierarchy."""
    import mujoco

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    m = mujoco.MjModel.from_xml_path(model_path())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # home pose world transforms

    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)

    # Node frames (world, at home).
    head_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "head")
    frame = {
        "base": np.eye(4),
        "body": _mat(d.xpos[bid("body_down_3dprint")], d.xmat[bid("body_down_3dprint")]),
        "head": _mat(d.site_xpos[head_site], np.eye(3)),  # control frame: world-aligned
        "antenna_r": _mat(d.xpos[bid("dc15_a01_horn_dummy_7")], d.xmat[bid("dc15_a01_horn_dummy_7")]),
        "antenna_l": _mat(d.xpos[bid("dc15_a01_horn_dummy_8")], d.xmat[bid("dc15_a01_horn_dummy_8")]),
    }

    # body id -> group
    body2group = {bid(bn): g for g, bns in GROUPS.items() for bn in bns}

    scene = trimesh.Scene()
    # Register the animatable group frames (named nodes three.js will drive). Each is a
    # pure transform node; meshes hang off it as separate children.
    for group in GROUPS:
        parent = PARENT[group]
        rest = frame[group] if parent is None else np.linalg.inv(frame[parent]) @ frame[group]
        scene.graph.update(
            frame_to=group,
            frame_from=parent if parent else scene.graph.base_frame,
            matrix=rest,
        )

    # Add each geom as its OWN mesh under its group frame — NOT merged. Merging parts
    # and recomputing normals across them smeared/inverted normals at part boundaries,
    # which is what looked splotchy. Per-part fix_normals() gives consistent winding.
    for group in GROUPS:
        finv = np.linalg.inv(frame[group])
        n = 0
        for gi in range(m.ngeom):
            if body2group.get(int(m.geom_bodyid[gi])) != group:
                continue
            gm = _geom_mesh(m, d, gi)
            if gm is None:
                continue
            gm.fix_normals()         # consistent outward winding (no dark inverted faces)
            gm.apply_transform(finv)  # world -> group-local
            scene.add_geometry(
                gm, node_name=f"{group}__{n}", parent_node_name=group, transform=np.eye(4)
            )
            n += 1

    glb = trimesh.exchange.gltf.export_glb(scene, include_normals=True)
    out_path.write_bytes(glb)
    return out_path


if __name__ == "__main__":
    p = export_glb()
    print("wrote", p, p.stat().st_size // 1024, "KB")
    # verify node names round-trip
    s = trimesh.load(p)
    print("nodes:", [n for n in s.graph.nodes_geometry])
