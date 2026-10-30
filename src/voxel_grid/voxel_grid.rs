use std::collections::{BinaryHeap, HashMap};

use bit_set::BitSet;
use nalgebra::SimdPartialOrd;

use crate::{
    math::{Aabb, AabbCellIter, FVec3, IDistanceHeuristic, IVec3},
    voxel_grid::{
        cell_identifiers::{CellID, CellPosition, ChunkID, ChunkPosition},
        chunks::{Chunk, ChunkSize, ChunkState},
    },
};

#[derive(Debug, Clone)]
pub struct VoxelGrid<C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> {
    chunks: Vec<Chunk<C, S, T>>,
    initialized_chunks: BitSet<u64>,
    cell_dimensions: IVec3,
    chunk_dimensions: IVec3,
}

impl<C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> VoxelGrid<C, S, T> {
    pub const fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub const fn cell_count(&self) -> usize {
        self.chunk_count() * Chunk::<C, S>::SIZE
    }

    pub const fn contains_cell(&self, cell: CellID<S>) -> bool {
        cell.global_index() < self.cell_count()
    }

    pub const fn contains_chunk(&self, chunk: ChunkID) -> bool {
        chunk.index() < self.chunk_count()
    }

    pub const fn cell_dimensions(&self) -> IVec3 {
        self.cell_dimensions
    }

    pub const fn chunk_dimensions(&self) -> IVec3 {
        self.chunk_dimensions
    }

    pub fn is_chunk_init(&self, chunk: ChunkID) -> bool {
        self.initialized_chunks.contains(chunk.index())
    }

    /// resize this nav grid with the given dimensions, potentially "cropping" out-of-bounds chunks
    ///
    ///  - new_chunk_dimensions - the new dimensions
    ///  - offset - the position of the original grid relative to the new grid
    ///
    ///  Note: after resizing, saved CellIDs are almost assuredly invalid or reference different cells
    pub fn resize(&mut self, new_chunk_dimensions: IVec3, offset: IVec3) {
        assert!(
            new_chunk_dimensions.x > 0 && new_chunk_dimensions.y > 0 && new_chunk_dimensions.z > 0
        );

        let new_chunk_count =
            (new_chunk_dimensions.x * new_chunk_dimensions.y * new_chunk_dimensions.z) as usize;
        let mut new_chunks = Vec::from_iter(std::iter::repeat_n(Chunk::default(), new_chunk_count));
        let mut new_initialized_chunks = BitSet::with_capacity_general(new_chunk_count);

        let old_chunk_dims = self.chunk_dimensions;
        let old_chunks = std::mem::take(&mut self.chunks);
        let old_init = std::mem::replace(&mut self.initialized_chunks, BitSet::new_general());

        for (old_index, chunk) in old_chunks.into_iter().enumerate() {
            let x = (old_index as isize) % old_chunk_dims.x;
            let temp_yz = (old_index as isize) / old_chunk_dims.x;
            let y = temp_yz % old_chunk_dims.y;
            let z = temp_yz / old_chunk_dims.y;

            let old_pos = IVec3::new(x, y, z);
            let new_pos = old_pos + offset.coords;

            if new_pos.x >= 0
                && new_pos.y >= 0
                && new_pos.z >= 0
                && new_pos.x < new_chunk_dimensions.x
                && new_pos.y < new_chunk_dimensions.y
                && new_pos.z < new_chunk_dimensions.z
            {
                let new_index = (new_pos.x
                    + new_chunk_dimensions.x * (new_pos.y + new_chunk_dimensions.y * new_pos.z))
                    as usize;
                new_chunks[new_index] = chunk;
                if old_init.contains(old_index) {
                    new_initialized_chunks.insert(new_index);
                }
            }
        }

        self.chunks = new_chunks;
        self.initialized_chunks = new_initialized_chunks;
        self.chunk_dimensions = new_chunk_dimensions;
        self.cell_dimensions = IVec3::new(
            new_chunk_dimensions.x * Chunk::<C, S>::SIDE_LENGTH as isize,
            new_chunk_dimensions.y * Chunk::<C, S>::SIDE_LENGTH as isize,
            new_chunk_dimensions.z * Chunk::<C, S>::SIDE_LENGTH as isize,
        );
    }

    pub fn cell_face_connective_neighbors(&self, cell: CellID<S>) -> Vec<CellID<S>> {
        if let Some(position) = self.get_cell_position(cell) {
            let mut output = Vec::with_capacity(6);

            const DELTAS: [nalgebra::Vector3<isize>; 6] = [
                nalgebra::Vector3::new(-1, 0, 0),
                nalgebra::Vector3::new(1, 0, 0),
                nalgebra::Vector3::new(0, -1, 0),
                nalgebra::Vector3::new(0, 1, 0),
                nalgebra::Vector3::new(0, 0, -1),
                nalgebra::Vector3::new(0, 0, 1),
            ];

            for d in DELTAS {
                if let Some(id) = self.get_cell_id(CellPosition(position.0 + d)) {
                    output.push(id)
                }
            }

            output
        } else {
            Vec::with_capacity(0)
        }
    }

    /// constructs a new empty navgrid from the given dimensions
    /// panics if cell dimensions are negative
    pub fn from_cell_dimensions(cell_dimensions: IVec3) -> Self {
        assert!(cell_dimensions.x > 0 && cell_dimensions.y > 0 && cell_dimensions.z > 0);

        if cell_dimensions.x == 0 || cell_dimensions.y == 0 || cell_dimensions.z == 0 {
            return Self {
                chunks: Vec::new(),
                initialized_chunks: BitSet::new_general(),
                cell_dimensions: cell_dimensions,
                chunk_dimensions: IVec3::new(0, 0, 0),
            };
        }

        let chunk_dimensions = IVec3::new(
            (cell_dimensions.x as usize).div_ceil(Chunk::<C, S>::SIDE_LENGTH) as isize,
            (cell_dimensions.y as usize).div_ceil(Chunk::<C, S>::SIDE_LENGTH) as isize,
            (cell_dimensions.z as usize).div_ceil(Chunk::<C, S>::SIDE_LENGTH) as isize,
        );
        let chunk_count: usize =
            (chunk_dimensions.x * chunk_dimensions.y * chunk_dimensions.z) as usize;

        Self {
            chunks: Vec::from_iter(std::iter::repeat_n(Chunk::default(), chunk_count)),
            initialized_chunks: BitSet::with_capacity_general(chunk_count),
            cell_dimensions,
            chunk_dimensions,
        }
    }

    /// gets the id of the given chunk
    /// out of bounds positions return None
    /// null chunks are never returned
    pub fn get_chunk_id(&self, pos: ChunkPosition) -> Option<ChunkID> {
        ChunkID::try_from_position_and_dimensions(pos, self.cell_dimensions)
    }

    /// gets the position of the containing chunk
    pub fn get_chunk_position(&self, id: ChunkID) -> Option<ChunkPosition> {
        if id.index() >= self.chunks.len() {
            return None;
        }

        let chunk_index = id.index();
        let chunk_pos_x = chunk_index % self.chunk_dimensions.x as usize;
        let temp_yz = chunk_index / self.chunk_dimensions.x as usize;

        let chunk_pos_y = temp_yz % self.chunk_dimensions.y as usize;
        let chunk_pos_z = temp_yz / self.chunk_dimensions.y as usize;

        Some(ChunkPosition(IVec3::new(
            chunk_pos_x as isize,
            chunk_pos_y as isize,
            chunk_pos_z as isize,
        )))
    }

    /// gets the cell position as an option
    /// out of bounds positions return None
    /// null cells are never returned
    pub fn get_cell_id(&self, pos: CellPosition) -> Option<CellID<S>> {
        if pos.x >= self.cell_dimensions.x
            || pos.y >= self.cell_dimensions.y
            || pos.z >= self.cell_dimensions.z
            || pos.x < 0
            || pos.y < 0
            || pos.z < 0
        {
            return None;
        }

        let chunk_pos = IVec3::new(
            pos.x / Chunk::<C, S>::SIDE_LENGTH as isize,
            pos.y / Chunk::<C, S>::SIDE_LENGTH as isize,
            pos.z / Chunk::<C, S>::SIDE_LENGTH as isize,
        );

        Some(CellID::new(
            unsafe {
                self.get_chunk_id(ChunkPosition(chunk_pos))
                    .unwrap_unchecked()
            },
            pos.local_position::<S>().into(),
        ))
    }

    /// gets the position of the cell
    pub fn get_cell_position(&self, id: CellID<S>) -> Option<CellPosition> {
        if !self.contains_cell(id) {
            return None;
        }

        let chunk_pos = unsafe { self.get_chunk_position(id.chunk_id()).unwrap_unchecked() };
        let local_position = id.local_position();

        Some(CellPosition(
            chunk_pos.0 * Chunk::<C, S>::SIDE_LENGTH as isize + local_position.0.coords,
        ))
    }

    /// gets the chunk by id
    pub fn get_chunk(&mut self, id: ChunkID) -> Option<&Chunk<C, S, T>> {
        self.chunks.get(id.index())
    }

    /// accesses the chunk mutably by id
    /// returns the return value of the mutator if it is run, otherwise None
    ///
    /// panics for invalid ChunkID
    pub fn get_chunk_mut<U>(
        &mut self,
        id: ChunkID,
        mutator: impl FnOnce(&mut Chunk<C, S, T>) -> U,
    ) -> Option<U> {
        if let Some(chunk) = self.chunks.get_mut(id.index()) {
            let out = mutator(chunk);
            self.initialized_chunks
                .get_mut()
                .set(id.index(), chunk.is_init());
            Some(out)
        } else {
            None
        }
    }

    /// gets the cell by id
    pub fn get_cell(&self, id: CellID<S>) -> Option<C> {
        let chunk = self.chunks.get(id.chunk_id().index())?;
        Some(chunk.get_cell(id.local_index()))
    }

    /// gets the cell mutably by id
    /// panics if the cell's containing chunk is not yet initialized
    pub fn get_cell_mut<U>(
        &mut self,
        id: CellID<S>,
        mutator: impl FnOnce(usize, &mut C, &mut T) -> U,
    ) -> Option<U> {
        self.get_chunk_mut(id.chunk_id(), |chunk| {
            chunk.get_cell_mut(id.local_index(), mutator)
        })
    }

    /// runs astar between two points, with the given predicate indicating whether a cell is traversible
    /// uses manhattan distance
    ///
    /// returns
    ///  - a path containing the end point of a path exists
    ///  - a path not containing the endpoint if a path does not exist,
    ///    this path instead returns the reachable point nearest to the end
    ///  - an empty path if the start or end cells are out of bounds or not traversible
    pub fn astar<D: IDistanceHeuristic>(
        &self,
        start: CellID<S>,
        end: CellID<S>,
        mut max_explored: usize,
        mut is_occupied: impl FnMut(CellID<S>, C) -> bool,
    ) -> Vec<CellID<S>> {
        fn reconstruct_path<S: ChunkSize>(
            predecessor_map: HashMap<CellID<S>, CellID<S>>,
            mut current: CellID<S>,
        ) -> Vec<CellID<S>> {
            let mut path = vec![current];

            while let Some(pre) = predecessor_map.get(&current) {
                current = *pre;
                path.push(current);
            }

            path.reverse();
            path
        }

        // validate max explored
        if max_explored <= 0 {
            return Vec::with_capacity(0);
        }

        // validate start and end cells
        {
            let start_state = match self.get_cell(start) {
                None => return Vec::with_capacity(0),
                Some(state) => state,
            };
            if is_occupied(start, start_state) {
                return Vec::with_capacity(0);
            }

            let end_state = match self.get_cell(end) {
                None => return Vec::with_capacity(0),
                Some(state) => state,
            };
            if is_occupied(end, end_state) {
                return Vec::with_capacity(0);
            }
        }

        /// comparisons eq/partial_eq and cmp/partial_cmp are based on fscore only
        #[derive(Debug, Clone, Copy)]
        struct AStarCell<S: ChunkSize> {
            cell: CellID<S>,
            fscore: u64,
        }
        impl<S: ChunkSize> PartialEq for AStarCell<S> {
            fn eq(&self, other: &Self) -> bool {
                self.fscore == other.fscore
            }
        }
        impl<S: ChunkSize> Eq for AStarCell<S> {}
        impl<S: ChunkSize> PartialOrd for AStarCell<S> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }
        impl<S: ChunkSize> Ord for AStarCell<S> {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                other.fscore.cmp(&self.fscore)
            }
        }

        // the position of the end cell
        let end_position = self.get_cell_position(end).unwrap().0;

        // the set of all cells open for exploration
        let mut open_set = BinaryHeap::new();

        // a map from a cell to the cell that discovered it
        let mut predecessor_map: HashMap<CellID<S>, CellID<S>> = HashMap::new();

        // a map from a cell to its gscore (weight of path to start)
        let mut gscore_map: HashMap<CellID<S>, u32> = HashMap::new();

        // the cell with the greatest fscore
        let mut nearest = start;
        let mut nearest_fscore = D::apply(
            self.get_cell_position(start).unwrap().0,
            self.get_cell_position(end).unwrap().0,
        );

        open_set.push(AStarCell {
            cell: start,
            fscore: nearest_fscore,
        });
        gscore_map.insert(start, 0);

        while let Some(current) = open_set.pop() {
            if current.cell == end {
                return reconstruct_path(predecessor_map, current.cell);
            }

            'neighbor: for neighbor in self.cell_face_connective_neighbors(current.cell) {
                if is_occupied(neighbor, self.get_cell(neighbor).unwrap()) {
                    continue 'neighbor;
                }

                let neighbor_position = self.get_cell_position(neighbor).unwrap().0;

                let temp_gscore = gscore_map[&current.cell] + 1;
                if temp_gscore < gscore_map.get(&neighbor).copied().unwrap_or(u32::MAX) {
                    if predecessor_map.insert(neighbor, current.cell).is_none() {
                        max_explored -= 1;
                    }
                    gscore_map.insert(neighbor, temp_gscore);

                    let fscore = temp_gscore as u64 + D::apply(neighbor_position, end_position);

                    open_set.push(AStarCell {
                        cell: neighbor,
                        fscore,
                    });

                    if fscore < nearest_fscore {
                        nearest = neighbor;
                        nearest_fscore = fscore;
                    }
                }

                if max_explored <= 0 {
                    return reconstruct_path(predecessor_map, nearest);
                }
            }
        }

        reconstruct_path(predecessor_map, nearest)
    }

    /// runs astar between two points, with the given predicate indicating whether a cell is traversible
    /// uses manhattan distance for astar
    ///
    /// then uses Simple Stupid Funnel Algorithm for path smoothing
    ///
    /// returns
    ///  - a path containing the end point of a path exists
    ///  - a path not containing the endpoint if a path does not exist,
    ///    this path instead returns the reachable point nearest to the end
    ///  - an empty path if the start or end cells are out of bounds or not traversible
    pub fn funnelled_astar<D: IDistanceHeuristic>(
        &self,
        start: CellID<S>,
        end: CellID<S>,
        max_explored: usize,
        mut is_occupied: impl FnMut(CellID<S>, C) -> bool,
    ) -> Vec<CellID<S>> {
        let path = self.astar::<D>(start, end, max_explored, &mut is_occupied);
        if path.len() <= 2 {
            return path;
        }

        // Helper to check line of sight between two cell IDs
        let mut has_line_of_sight = |c1: CellID<S>, c2: CellID<S>| -> bool {
            let p1 = self.get_cell_position(c1);
            let p2 = self.get_cell_position(c2);
            if p1.is_none() || p2.is_none() {
                return false;
            }
            let f1 = FVec3::new(
                p1.unwrap().x as f32 + 0.5,
                p1.unwrap().y as f32 + 0.5,
                p1.unwrap().z as f32 + 0.5,
            );
            let f2 = FVec3::new(
                p2.unwrap().x as f32 + 0.5,
                p2.unwrap().y as f32 + 0.5,
                p2.unwrap().z as f32 + 0.5,
            );

            // Simplified check: raymarch or check points along segment
            let steps = (nalgebra::distance(&f1, &f2).ceil() as usize).max(1);
            for i in 0..=steps {
                let t = i as f32 / steps as f32;
                let sample_pos = FVec3::new(
                    f1.x + t * (f2.x - f1.x),
                    f1.y + t * (f2.y - f1.y),
                    f1.z + t * (f2.z - f1.z),
                );
                let cell_pos = CellPosition(IVec3::new(
                    sample_pos.x.floor() as isize,
                    sample_pos.y.floor() as isize,
                    sample_pos.z.floor() as isize,
                ));
                if let Some(id) = self.get_cell_id(cell_pos) {
                    if let Some(val) = self.get_cell(id) {
                        if is_occupied(id, val) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            true
        };

        // Smooth path using shortcutting / string pulling
        let mut smoothed = Vec::new();
        let mut i = 0;
        smoothed.push(path[0]);

        while i < path.len() - 1 {
            let mut furthest = i + 1;
            for j in (i + 2)..path.len() {
                if has_line_of_sight(path[i], path[j]) {
                    furthest = j;
                }
            }
            smoothed.push(path[furthest]);
            i = furthest;
        }

        smoothed
    }

    /// gets an iterator over chunk ids in the given chunk aabb
    /// clamping the given aabb to valid boundary
    pub fn aabb_chunk_index_iter(&self, aabb: Aabb<ChunkPosition>) -> AabbChunkIDIter {
        let aabb = Aabb {
            min: aabb.min.simd_max(IVec3::new(0, 0, 0)),
            max: aabb.max.simd_min(self.chunk_dimensions),
        };

        AabbChunkIDIter {
            cell_iter: AabbCellIter::new(Aabb::new(
                aabb.min.simd_min(IVec3::new(0, 0, 0)),
                aabb.max.simd_max(self.chunk_dimensions),
            )),
        }
    }

    pub fn aabb_chunk_iter(
        &self,
        aabb: Aabb<ChunkPosition>,
    ) -> VoxelChunkIter<'_, C, S, T, AabbChunkIDIter> {
        VoxelChunkIter {
            chunks: &self.chunks,
            id_iter: self.aabb_chunk_index_iter(aabb),
        }
    }

    pub fn aabb_chunk_initializing_iter_mut(
        &mut self,
        aabb: Aabb<ChunkPosition>,
    ) -> VoxelChunkInitializingIterMut<'_, C, S, T, AabbChunkIDIter> {
        let id_iter = self.aabb_chunk_index_iter(aabb);
        VoxelChunkInitializingIterMut {
            grid: self,
            id_iter,
        }
    }

    pub fn aabb_chunk_non_initializing_iter_mut(
        &mut self,
        aabb: Aabb<ChunkPosition>,
    ) -> VoxelChunkNonInitializingIterMut<'_, C, S, T, AabbChunkIDIter> {
        let id_iter = self.aabb_chunk_index_iter(aabb);
        VoxelChunkNonInitializingIterMut {
            grid: self,
            id_iter,
        }
    }
}

/// an iterator over an aabb of chunks
#[derive(Debug, Clone)]
pub struct AabbChunkIDIter {
    cell_iter: AabbCellIter,
}

impl Iterator for AabbChunkIDIter {
    type Item = ChunkID;

    fn next(&mut self) -> Option<Self::Item> {
        self.cell_iter.next().map(|pos| {
            ChunkID::try_from_position_and_dimensions(ChunkPosition(pos), self.cell_iter.end())
                .unwrap()
        })
    }
}

/// a mutable iterator over chunks in a voxel map
/// uninitialized chunks are skipped
#[derive(Debug)]
pub struct VoxelChunkNonInitializingIterMut<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = ChunkID>,
> {
    grid: &'a mut VoxelGrid<C, S, T>,
    id_iter: I,
}

impl<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = ChunkID>,
> Iterator for VoxelChunkNonInitializingIterMut<'a, C, S, T, I>
{
    type Item = (ChunkID, &'a mut Chunk<C, S, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut id = self.id_iter.next()?;
        while !self.grid.is_chunk_init(id) {
            id = self.id_iter.next()?;
        }
        let chunk_ptr = std::ptr::from_mut(&mut self.grid.chunks[id.index()]);
        Some(unsafe { (id, chunk_ptr.as_mut_unchecked()) })
    }
}

/// a mutable iterator over chunks in a voxel map
/// chunks are initialized before being returned
#[derive(Debug)]
pub struct VoxelChunkInitializingIterMut<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = ChunkID>,
> {
    grid: &'a mut VoxelGrid<C, S, T>,
    id_iter: I,
}

impl<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = ChunkID>,
> Iterator for VoxelChunkInitializingIterMut<'a, C, S, T, I>
{
    type Item = (ChunkID, &'a mut Chunk<C, S, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.id_iter.next()?;
        self.grid.chunks[id.index()].init();
        self.grid.initialized_chunks.insert(id.index());
        let chunk_ptr = std::ptr::from_mut(&mut self.grid.chunks[id.index()]);
        Some(unsafe { (ChunkID::new(id.index()), chunk_ptr.as_mut_unchecked()) })
    }
}

/// an iterator over chunks in a voxel map
#[derive(Debug, Clone)]
pub struct VoxelChunkIter<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = ChunkID>,
> {
    chunks: &'a [Chunk<C, S, T>],
    id_iter: I,
}

impl<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = ChunkID>,
> Iterator for VoxelChunkIter<'a, C, S, T, I>
{
    type Item = (ChunkID, &'a Chunk<C, S, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.id_iter.next()?;
        Some((id, unsafe { self.chunks.get_unchecked(id.index()) }))
    }
}
