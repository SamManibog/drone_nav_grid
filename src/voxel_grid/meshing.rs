use std::{
    collections::HashMap,
    fs::File,
    io::{self, BufWriter, Write},
    path::Path,
};

use bit_set::BitSet;

use crate::voxel_grid::{
    cell_identifiers::{CellID, ChunkID},
    chunks::{ChunkSize, ChunkState},
    voxel_grid::VoxelGrid,
};

/// face normals, indexed by `Quad::normal` (axis * 2 + (1 if negative direction))
const NORMALS: [[i8; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

/// a rectangular face, wound counter-clockwise when viewed from the side it faces
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Quad {
    /// indices into `QuadMesh::vertices`
    pub vertices: [u32; 4],
    /// index into the six axis-aligned normals: +x, -x, +y, -y, +z, -z
    pub normal: u8,
}

/// a mesh of axis-aligned quads, in cell coordinates (cell (i, j, k) spans [i, i+1) per axis)
#[derive(Debug, Clone, Default)]
pub struct QuadMesh {
    pub vertices: Vec<[isize; 3]>,
    pub quads: Vec<Quad>,
}

impl QuadMesh {
    /// writes the mesh in Wavefront .obj format (quads, shared vertices, flat normals)
    pub fn write_obj(&self, w: &mut impl Write) -> io::Result<()> {
        writeln!(
            w,
            "# voxel mesh: {} vertices, {} quads",
            self.vertices.len(),
            self.quads.len()
        )?;
        for v in &self.vertices {
            writeln!(w, "v {} {} {}", v[0], v[1], v[2])?;
        }
        for n in NORMALS {
            writeln!(w, "vn {} {} {}", n[0], n[1], n[2])?;
        }
        for q in &self.quads {
            let n = q.normal as usize + 1;
            let [a, b, c, d] = q.vertices.map(|i| i as usize + 1);
            writeln!(w, "f {a}//{n} {b}//{n} {c}//{n} {d}//{n}")?;
        }
        Ok(())
    }
}

fn vertex_index(
    map: &mut HashMap<[isize; 3], u32>,
    vertices: &mut Vec<[isize; 3]>,
    p: [isize; 3],
) -> u32 {
    *map.entry(p).or_insert_with(|| {
        vertices.push(p);
        (vertices.len() - 1) as u32
    })
}

impl<C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> VoxelGrid<C, S, T> {
    /// builds a face-culled, greedy-meshed quad mesh of every cell satisfying `predicate`.
    ///
    ///  - `predicate(cell id, cell state, containing chunk's user state)`
    ///  - only faces between a matching cell and a non-matching cell (or the grid boundary)
    ///    are emitted, and coplanar adjacent faces are merged into maximal rectangles
    ///  - uninitialized chunks are skipped (treated as containing no matching cells)
    ///  - the predicate may also be called on padding cells past `cell_dimensions()` in edge
    ///    chunks; those are never meshed
    pub fn build_quad_mesh(&self, predicate: impl Fn(CellID<S>, C, &T) -> bool) -> QuadMesh {
        let mut mesh = QuadMesh::default();
        if self.chunk_count() == 0 {
            return mesh;
        }

        let cell_dims = self.cell_dimensions();
        let chunk_dims = self.chunk_dimensions();
        let dims = [cell_dims.x, cell_dims.y, cell_dims.z];
        let cdims = [chunk_dims.x, chunk_dims.y, chunk_dims.z];
        let shift = S::POWER_OF_TWO;
        let side = S::SIDE_LENGTH as isize;
        let local_mask = side - 1;

        // per-chunk set of matching cells; None = uninitialized or nothing matched
        let masks: Vec<Option<BitSet<u64>>> = (0..self.chunk_count())
            .map(|i| {
                let id = ChunkID::new(i);
                self.get_chunk(id)?
                    .mask_where(|local, cell, state| predicate(CellID::new(id, local), cell, state))
            })
            .collect();

        let chunk_index = |cp: [isize; 3]| (cp[0] + cdims[0] * (cp[1] + cdims[1] * cp[2])) as usize;

        let solid = |p: [isize; 3]| -> bool {
            if p[0] < 0
                || p[1] < 0
                || p[2] < 0
                || p[0] >= dims[0]
                || p[1] >= dims[1]
                || p[2] >= dims[2]
            {
                return false;
            }
            match &masks[chunk_index([p[0] >> shift, p[1] >> shift, p[2] >> shift])] {
                None => false,
                Some(m) => {
                    let local = (p[0] & local_mask)
                        + side * ((p[1] & local_mask) + side * (p[2] & local_mask));
                    m.contains(local as usize)
                }
            }
        };

        let mut vertex_map: HashMap<[isize; 3], u32> = HashMap::new();

        for d in 0..3 {
            let ua = (d + 1) % 3; // (ua, va, d) is a right-handed frame, so ua x va = +d
            let va = (d + 2) % 3;
            let (du, dv) = (dims[ua] as usize, dims[va] as usize);

            // true = a face is present at (u, v). greedy merging clears every entry it
            // consumes, so the buffer is all-false again after each slice
            let mut face_mask = vec![false; du * dv];

            for negative in [false, true] {
                let step: isize = if negative { -1 } else { 1 };

                for s in 0..dims[d] {
                    let mut any = false;

                    // build the face mask one chunk tile at a time, skipping tiles with no matches
                    for cv in 0..cdims[va] {
                        for cu in 0..cdims[ua] {
                            let mut cp = [0isize; 3];
                            cp[d] = s >> shift;
                            cp[ua] = cu;
                            cp[va] = cv;
                            if masks[chunk_index(cp)].is_none() {
                                continue;
                            }

                            let u_range = (cu << shift)..(((cu + 1) << shift).min(dims[ua]));
                            let v_range = (cv << shift)..(((cv + 1) << shift).min(dims[va]));
                            for v in v_range {
                                for u in u_range.clone() {
                                    let mut p = [0isize; 3];
                                    p[d] = s;
                                    p[ua] = u;
                                    p[va] = v;
                                    let mut n = p;
                                    n[d] += step;
                                    if solid(p) && !solid(n) {
                                        face_mask[u as usize + v as usize * du] = true;
                                        any = true;
                                    }
                                }
                            }
                        }
                    }

                    if !any {
                        continue;
                    }

                    // greedy merge: grow right along u, then down along v
                    let plane = if negative { s } else { s + 1 };
                    for v in 0..dv {
                        let mut u = 0;
                        while u < du {
                            if !face_mask[u + v * du] {
                                u += 1;
                                continue;
                            }

                            let mut w = 1;
                            while u + w < du && face_mask[u + w + v * du] {
                                w += 1;
                            }

                            let mut h = 1;
                            'grow: while v + h < dv {
                                for k in 0..w {
                                    if !face_mask[u + k + (v + h) * du] {
                                        break 'grow;
                                    }
                                }
                                h += 1;
                            }

                            for dy in 0..h {
                                for dx in 0..w {
                                    face_mask[u + dx + (v + dy) * du] = false;
                                }
                            }

                            let corner = |uu: usize, vv: usize| {
                                let mut p = [0isize; 3];
                                p[d] = plane;
                                p[ua] = uu as isize;
                                p[va] = vv as isize;
                                p
                            };
                            let corners = [
                                corner(u, v),
                                corner(u + w, v),
                                corner(u + w, v + h),
                                corner(u, v + h),
                            ];
                            // counter-clockwise seen from the face's outside
                            let order: [usize; 4] = if negative { [0, 3, 2, 1] } else { [0, 1, 2, 3] };

                            let vertices = order.map(|i| {
                                vertex_index(&mut vertex_map, &mut mesh.vertices, corners[i])
                            });
                            mesh.quads.push(Quad {
                                vertices,
                                normal: (d * 2 + negative as usize) as u8,
                            });

                            u += w;
                        }
                    }
                }
            }
        }

        mesh
    }

    /// writes a Wavefront .obj mesh of every cell satisfying `predicate` to `path`.
    /// see `build_quad_mesh` for the meshing rules. Vertices are in cell coordinates;
    /// scale/translate on import (or map through your grid's transform) for world units.
    pub fn export_obj(
        &self,
        predicate: impl Fn(CellID<S>, C, &T) -> bool,
        path: impl AsRef<Path>,
    ) -> io::Result<()> {
        let mesh = self.build_quad_mesh(predicate);
        let mut writer = BufWriter::new(File::create(path)?);
        mesh.write_obj(&mut writer)?;
        writer.flush()
    }
}
