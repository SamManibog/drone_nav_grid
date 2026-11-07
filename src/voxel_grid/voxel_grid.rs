use std::{
    cell::Cell,
    collections::{BinaryHeap, HashMap, VecDeque},
    marker::PhantomData,
};

use bit_set::BitSet;
use nalgebra::SimdPartialOrd;

use crate::{
    math::{
        Aabb, AabbCellIter, FVec3, Grid, GridExt, GridSegmentOverlapIter, IDistanceHeuristic,
        IVec3, LineSegment,
    },
    voxel_grid::{
        cell_identifiers::{CellID, CellPosition, ChunkID, ChunkPosition, LocalCellPosition},
        chunks::{AabbIndexIter, Chunk, ChunkSize, ChunkState, InitializedChunk},
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
        assert!(cell_dimensions.x >= 0 && cell_dimensions.y >= 0 && cell_dimensions.z >= 0);

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
        ChunkID::try_from_position_and_dimensions(pos, self.chunk_dimensions)
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
    pub fn get_chunk(&self, id: ChunkID) -> Option<&Chunk<C, S, T>> {
        self.chunks.get(id.index())
    }

    /// accesses the chunk mutably by id
    /// returns the return value of the mutator if it is run, otherwise None
    pub fn get_chunk_mut<U>(
        &mut self,
        id: ChunkID,
        mutator: impl FnOnce(&mut Chunk<C, S, T>) -> U,
    ) -> Option<U> {
        if let Some(chunk) = self.chunks.get_mut(id.index()) {
            let out = mutator(chunk);
            if chunk.is_init() {
                self.initialized_chunks.insert(id.index());
            } else {
                self.initialized_chunks.remove(id.index());
            }
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
    ///
    /// returns (vec, bool)
    ///  - a path containing the end point of a path exists, and true
    ///  - a path not containing the endpoint if a path does not exist,
    ///    this path instead returns the reachable point nearest to the end, and false
    ///  - an empty path if the start or end cells are out of bounds or not traversible, and false
    pub fn astar<D: IDistanceHeuristic>(
        &self,
        start: CellID<S>,
        end: CellID<S>,
        mut max_explored: usize,
        mut is_occupied: impl FnMut(CellID<S>, C) -> bool,
    ) -> (Vec<CellID<S>>, bool) {
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
            return (Vec::with_capacity(0), false);
        }

        // validate start and end cells
        {
            let start_state = match self.get_cell(start) {
                None => return (Vec::with_capacity(0), false),
                Some(state) => state,
            };
            if is_occupied(start, start_state) {
                return (Vec::with_capacity(0), false);
            }

            let end_state = match self.get_cell(end) {
                None => return (Vec::with_capacity(0), false),
                Some(state) => state,
            };
            if is_occupied(end, end_state) {
                return (Vec::with_capacity(0), false);
            }
        }

        /// comparisons eq/partial_eq and cmp/partial_cmp are based on fscore only
        #[derive(Debug, Clone, Copy)]
        struct AStarCell<S: ChunkSize> {
            cell: CellID<S>,
            fscore: u64,
            /// gscore at push time, used to skip stale heap entries
            gscore: u32,
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
        let mut nearest_heuristic = D::apply(
            self.get_cell_position(start).unwrap().0,
            self.get_cell_position(end).unwrap().0,
        );

        open_set.push(AStarCell {
            cell: start,
            fscore: nearest_heuristic,
            gscore: 0,
        });
        gscore_map.insert(start, 0);

        while let Some(current) = open_set.pop() {
            // skip entries superseded by a cheaper route to the same cell
            if current.gscore > gscore_map[&current.cell] {
                continue;
            }

            if current.cell == end {
                return (reconstruct_path(predecessor_map, current.cell), true);
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

                    let distance_heuristic = D::apply(neighbor_position, end_position);
                    let fscore = temp_gscore as u64 + distance_heuristic;

                    open_set.push(AStarCell {
                        cell: neighbor,
                        fscore,
                        gscore: temp_gscore,
                    });

                    if distance_heuristic < nearest_heuristic {
                        nearest = neighbor;
                        nearest_heuristic = distance_heuristic;
                    }
                }

                if max_explored <= 0 {
                    return (reconstruct_path(predecessor_map, nearest), false);
                }
            }
        }

        (reconstruct_path(predecessor_map, nearest), false)
    }

    /// runs astar between two points, with the given predicate indicating whether a cell is traversible
    ///
    /// then uses greedy string pulling for path smoothing
    ///
    /// returns (vec, bool)
    ///  - a path containing the end point of a path exists, true
    ///  - a path not containing the endpoint if a path does not exist,
    ///    this path instead returns the reachable point nearest to the end, false
    ///  - an empty path if the start or end cells are out of bounds or not traversible, false
    pub fn funnelled_astar<D: IDistanceHeuristic>(
        &self,
        start: CellID<S>,
        end: CellID<S>,
        max_explored: usize,
        mut is_occupied: impl FnMut(CellID<S>, C) -> bool,
    ) -> (Vec<CellID<S>>, bool) {
        let (path, success) = self.astar::<D>(start, end, max_explored, &mut is_occupied);
        if path.len() <= 2 {
            return (path, success);
        }

        let view = VoxelCellView::new(self); // lives across all LOS queries so the cache carries over

        let mut has_line_of_sight = |c1: CellID<S>, c2: CellID<S>| -> bool {
            let (Some(p1), Some(p2)) = (self.get_cell_position(c1), self.get_cell_position(c2))
            else {
                return false;
            };
            let center =
                |p: CellPosition| FVec3::new(p.x as f32 + 0.5, p.y as f32 + 0.5, p.z as f32 + 0.5);
            let seg = LineSegment {
                start: center(p1),
                end: center(p2),
            };

            (&view)
                .segment_overlap_iter(seg)
                .all(|(id, value)| !is_occupied(id, value))
        };

        let mut smoothed = vec![path[0]];
        let mut i = 0;
        while i < path.len() - 1 {
            let mut next = i + 1;
            for j in ((i + 2)..path.len()).rev() {
                if has_line_of_sight(path[i], path[j]) {
                    next = j;
                    break;
                }
            }
            smoothed.push(path[next]);
            i = next;
        }

        (smoothed, success)
    }

    /// gets a breadth-first traversing iterator around the given cell
    /// uses manhattan distance
    pub fn breadth_first_iter(&self, cell: CellID<S>) -> BreadthFirstCellIDIter<'_, C, S, T> {
        let mut queue = VecDeque::new();

        if self.contains_cell(cell) {
            queue.push_back(cell);
        }

        BreadthFirstCellIDIter::new(self, queue)
    }

    /// gets an iterator over chunk ids in the given chunk aabb
    /// clamping the given aabb to valid boundary
    pub fn aabb_chunk_index_iter(&self, aabb: Aabb<ChunkPosition>) -> AabbChunkIDIter {
        let min = aabb.min.simd_max(IVec3::new(0, 0, 0));
        let max = aabb.max.simd_min(self.chunk_dimensions);
        AabbChunkIDIter {
            cell_iter: AabbCellIter::new(Aabb::new(min, max)),
            chunk_dimensions: self.chunk_dimensions,
        }
    }

    /// converts a segment in cell coordinates to chunk-lattice coordinates
    fn segment_to_chunk_space(segment: LineSegment) -> LineSegment {
        // side length is a power of two, so this scaling is exact
        let inv = 1.0 / S::SIDE_LENGTH as f32;
        LineSegment {
            start: FVec3::new(
                segment.start.x * inv,
                segment.start.y * inv,
                segment.start.z * inv,
            ),
            end: FVec3::new(
                segment.end.x * inv,
                segment.end.y * inv,
                segment.end.z * inv,
            ),
        }
    }

    /// gets an iterator over the ids of chunks overlapping the given line segment
    /// line segment is in cell coordinates
    pub fn segment_chunk_id_iter(
        &self,
        segment: LineSegment,
    ) -> GridSegmentOverlapIter<'static, ChunkIdGrid> {
        ChunkIdGrid {
            chunk_dimensions: self.chunk_dimensions,
        }
        .segment_overlap_iter(Self::segment_to_chunk_space(segment))
    }

    /// gets an iterator over chunks overlapping the given line segment
    /// line segment is in cell coordinates
    pub fn segment_chunk_iter(
        &self,
        segment: LineSegment,
    ) -> VoxelChunkIter<'_, C, S, T, GridSegmentOverlapIter<'static, ChunkIdGrid>> {
        VoxelChunkIter {
            chunks: &self.chunks,
            id_iter: self.segment_chunk_id_iter(segment),
        }
    }

    /// gets a mutable iterator over chunks overlapping the given line segment
    /// initializes any uninitialized chunks
    /// line segment is in cell coordinates
    pub fn segment_chunk_initializing_iter_mut(
        &mut self,
        segment: LineSegment,
    ) -> VoxelChunkInitializingIterMut<'_, C, S, T, GridSegmentOverlapIter<'static, ChunkIdGrid>>
    {
        let id_iter = self.segment_chunk_id_iter(segment);
        VoxelChunkInitializingIterMut {
            grid: self,
            id_iter,
        }
    }

    /// gets a mutable iterator over chunks overlapping the given line segment
    /// skips any uninitialized chunks
    /// line segment is in cell coordinates
    pub fn segment_chunk_skipping_iter_mut(
        &mut self,
        segment: LineSegment,
    ) -> VoxelChunkSkippingIterMut<'_, C, S, T, GridSegmentOverlapIter<'static, ChunkIdGrid>> {
        let id_iter = self.segment_chunk_id_iter(segment);
        VoxelChunkSkippingIterMut {
            grid: self,
            id_iter,
        }
    }

    /// gets an iterator over the ids of cells overlapping the given line segment
    /// line segment is in cell coordinates
    pub fn segment_cell_id_iter(
        &self,
        segment: LineSegment,
    ) -> GridSegmentOverlapIter<'static, CellIdGrid<S>> {
        CellIdGrid::<S> {
            cell_dimensions: self.cell_dimensions,
            chunk_dimensions: self.chunk_dimensions,
            marker: PhantomData,
        }
        .segment_overlap_iter(segment)
    }

    /// gets an iterator over cells overlapping the given line segment,
    /// yielding (cell id, cell state). Cells in uninitialized chunks yield C::default()
    /// line segment is in cell coordinates
    pub fn segment_cell_iter(
        &self,
        segment: LineSegment,
    ) -> VoxelCellIter<'_, C, S, T, GridSegmentOverlapIter<'static, CellIdGrid<S>>> {
        VoxelCellIter {
            cell_view: VoxelCellView::new(self),
            id_iter: self.segment_cell_id_iter(segment),
        }
    }

    /// calls `visit(cell id, cell state, containing chunk's user state)` for every cell overlapping
    /// the given line segment, in order along the segment.
    /// initializes any uninitialized chunks on the way.
    /// line segment is in cell coordinates
    ///
    /// visit should return true if iteration should continue and false if iteration should end
    ///
    /// This is internal iteration rather than an `Iterator` on purpose: yielding `&mut T` (the
    /// chunk state) for every cell of a chunk from an `Iterator` would hand out aliasing `&mut`s.
    pub fn segment_cell_initializing_for_each_mut(
        &mut self,
        segment: LineSegment,
        mut visit: impl FnMut(CellID<S>, &mut C, &mut T) -> bool,
    ) {
        // the id iterator owns copies of the dimensions, so it does not borrow self
        for id in self.segment_cell_id_iter(segment) {
            self.get_chunk_mut(id.chunk_id(), |chunk| {
                chunk.init();
                if !chunk.get_cell_mut(id.local_index(), |_, cell, state| visit(id, cell, state)) {
                    return;
                }
            });
        }
    }

    /// like `segment_cell_initializing_for_each_mut`, but cells in uninitialized chunks are skipped
    /// visit should return true if iteration should continue and false if iteration should end
    pub fn segment_cell_skipping_for_each(
        &self,
        segment: LineSegment,
        mut visit: impl FnMut(CellID<S>, C, &T) -> bool,
    ) {
        for id in self.segment_cell_id_iter(segment) {
            if !self.is_chunk_init(id.chunk_id()) {
                continue;
            }
            let chunk = self.get_chunk(id.chunk_id()).unwrap();
            let cell = chunk.get_cell(id.local_index());
            if !visit(id, cell, &chunk.user_state) {
                return;
            }
        }
    }

    /// like `segment_cell_initializing_for_each_mut`, but cells in uninitialized chunks are skipped
    /// visit should return true if iteration should continue and false if iteration should end
    pub fn segment_cell_skipping_for_each_mut(
        &mut self,
        segment: LineSegment,
        mut visit: impl FnMut(CellID<S>, &mut C, &mut T) -> bool,
    ) {
        for id in self.segment_cell_id_iter(segment) {
            if !self.is_chunk_init(id.chunk_id()) {
                continue;
            }
            self.get_chunk_mut(id.chunk_id(), |chunk| {
                if !chunk.get_cell_mut(id.local_index(), |_, cell, state| visit(id, cell, state)) {
                    return;
                }
            });
        }
    }

    /// calls `visit(cell position, cell state, containing chunk's user state)` for every cell in the
    /// given end-exclusive cell aabb (clamped to the grid), initializing chunks as needed
    pub fn aabb_cell_initializing_for_each_mut(
        &mut self,
        aabb: Aabb<IVec3>,
        mut visit: impl FnMut(CellPosition, &mut C, &mut T),
    ) {
        let bounds = Aabb::new(IVec3::new(0, 0, 0), self.cell_dimensions);
        let aabb = aabb.intersection(&bounds);
        if aabb.is_empty() {
            return;
        }

        let shift = S::POWER_OF_TWO;
        let side = S::SIDE_LENGTH as isize;
        let chunk_aabb = Aabb::new(
            ChunkPosition(IVec3::new(
                aabb.min.x >> shift,
                aabb.min.y >> shift,
                aabb.min.z >> shift,
            )),
            ChunkPosition(IVec3::new(
                ((aabb.max.x - 1) >> shift) + 1,
                ((aabb.max.y - 1) >> shift) + 1,
                ((aabb.max.z - 1) >> shift) + 1,
            )),
        );
        let chunk_dimensions = self.chunk_dimensions;

        for (id, mut chunk) in self.aabb_chunk_initializing_iter_mut(chunk_aabb) {
            let chunk_pos = ChunkPosition::try_from_id_and_dimension(id, chunk_dimensions).unwrap();
            let origin = chunk_pos.0.coords * side;

            let local_min = IVec3::new(
                (aabb.min.x - origin.x).max(0),
                (aabb.min.y - origin.y).max(0),
                (aabb.min.z - origin.z).max(0),
            );
            let local_max = IVec3::new(
                (aabb.max.x - origin.x).min(side),
                (aabb.max.y - origin.y).min(side),
                (aabb.max.z - origin.z).min(side),
            );

            for index in AabbIndexIter::<S>::new(
                LocalCellPosition::new(local_min),
                LocalCellPosition::new(local_max),
            ) {
                let local = LocalCellPosition::<S>::from_index(index);
                let pos = CellPosition(IVec3::from(origin + local.0.coords));
                chunk.get_cell_mut(index, |_, cell, state| visit(pos, cell, state));
            }
        }
    }

    /// gets an iterator over chunks
    pub fn aabb_chunk_iter(
        &self,
        aabb: Aabb<ChunkPosition>,
    ) -> VoxelChunkIter<'_, C, S, T, AabbChunkIDIter> {
        VoxelChunkIter {
            chunks: &self.chunks,
            id_iter: self.aabb_chunk_index_iter(aabb),
        }
    }

    /// gets a mutable iterator over chunks that initializes them before yielding them
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

    /// gets a mutable iterator over chunks that skips uninitialized chunks
    pub fn aabb_chunk_skipping_iter_mut(
        &mut self,
        aabb: Aabb<ChunkPosition>,
    ) -> VoxelChunkSkippingIterMut<'_, C, S, T, AabbChunkIDIter> {
        let id_iter = self.aabb_chunk_index_iter(aabb);
        VoxelChunkSkippingIterMut {
            grid: self,
            id_iter,
        }
    }
}

/// an iterator over an aabb of chunks
#[derive(Debug, Clone)]
pub struct AabbChunkIDIter {
    cell_iter: AabbCellIter,
    chunk_dimensions: IVec3,
}

impl Iterator for AabbChunkIDIter {
    type Item = ChunkID;

    fn next(&mut self) -> Option<Self::Item> {
        self.cell_iter.next().map(|pos| {
            ChunkID::try_from_position_and_dimensions(ChunkPosition(pos), self.chunk_dimensions)
                .unwrap()
        })
    }
}

/// a mutable iterator over chunks in a voxel map
/// uninitialized chunks are skipped
#[derive(Debug)]
pub struct VoxelChunkSkippingIterMut<
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
> Iterator for VoxelChunkSkippingIterMut<'a, C, S, T, I>
{
    type Item = (ChunkID, InitializedChunk<'a, C, S, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut id = self.id_iter.next()?;
        while !self.grid.is_chunk_init(id) {
            id = self.id_iter.next()?;
        }
        let chunk_ptr = std::ptr::from_mut(&mut self.grid.chunks[id.index()]);
        Some((
            id,
            unsafe { &mut *chunk_ptr }.borrow_as_initialized().unwrap(),
        ))
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
    type Item = (ChunkID, InitializedChunk<'a, C, S, T>);

    fn next(&mut self) -> Option<Self::Item> {
        let id = self.id_iter.next()?;
        self.grid.chunks[id.index()].init();
        self.grid.initialized_chunks.insert(id.index());
        let chunk_ptr = std::ptr::from_mut(&mut self.grid.chunks[id.index()]);
        Some((
            ChunkID::new(id.index()),
            unsafe { &mut *chunk_ptr }.borrow_as_initialized().unwrap(),
        ))
    }
}

/// an iterator over cells in a voxel map
/// caches the last visited chunk
pub struct VoxelCellIter<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = CellID<S>>,
> {
    cell_view: VoxelCellView<'a, C, S, T>,
    id_iter: I,
}

impl<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = CellID<S>>,
> Iterator for VoxelCellIter<'a, C, S, T, I>
{
    type Item = (CellID<S>, C);

    fn next(&mut self) -> Option<Self::Item> {
        let cell = self.id_iter.next()?;
        Some((cell, self.cell_view.lookup_by_id(cell).unwrap()))
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

/// the most recently visited chunk
struct CachedChunk<'g, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> {
    pos: IVec3,
    id: ChunkID,
    chunk: &'g Chunk<C, S, T>,
}

// manual impls: derive(Clone, Copy) would wrongly require T: Copy
impl<'g, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> Clone
    for CachedChunk<'g, C, S, T>
{
    fn clone(&self) -> Self {
        *self
    }
}
impl<'g, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> Copy
    for CachedChunk<'g, C, S, T>
{
}

/// read-only view of a grid for segment traversal.
/// caches the last chunk looked up, so cells along a segment
/// only pay the chunk lookup when the segment crosses a chunk boundary.
struct VoxelCellView<'g, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> {
    grid: &'g VoxelGrid<C, S, T>,
    last_chunk: Cell<Option<CachedChunk<'g, C, S, T>>>,
}

impl<'g, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>>
    VoxelCellView<'g, C, S, T>
{
    fn new(grid: &'g VoxelGrid<C, S, T>) -> Self {
        Self {
            grid,
            last_chunk: Cell::new(None),
        }
    }

    fn lookup_by_id(&self, id: CellID<S>) -> Option<C> {
        let chunk_id = id.chunk_id();

        let cached = match self.last_chunk.get() {
            Some(c) if c.id == chunk_id => c,
            _ => {
                let pos = self.grid.get_chunk_position(chunk_id)?;
                let c = CachedChunk {
                    pos: pos.0,
                    id: chunk_id,
                    chunk: self.grid.chunks.get(chunk_id.index())?,
                };
                self.last_chunk.set(Some(c));
                c
            }
        };

        // uninitialized chunks return C::default()
        Some(cached.chunk.get_cell(id.local_index()))
    }

    fn lookup(&self, position: IVec3) -> Option<(CellID<S>, C)> {
        let dims = self.grid.cell_dimensions;
        if position.x < 0
            || position.y < 0
            || position.z < 0
            || position.x >= dims.x
            || position.y >= dims.y
            || position.z >= dims.z
        {
            return None;
        }

        // side length is a power of two and position is non-negative
        let shift = S::POWER_OF_TWO;
        let mask = S::SIDE_LENGTH as isize - 1;

        let chunk_pos = IVec3::new(
            position.x >> shift,
            position.y >> shift,
            position.z >> shift,
        );

        let cached = match self.last_chunk.get() {
            Some(c) if c.pos == chunk_pos => c,
            _ => {
                let id = ChunkID::try_from_position_and_dimensions(
                    ChunkPosition(chunk_pos),
                    self.grid.chunk_dimensions,
                )?;
                let c = CachedChunk {
                    pos: chunk_pos,
                    id,
                    chunk: self.grid.chunks.get(id.index())?,
                };
                self.last_chunk.set(Some(c));
                c
            }
        };

        let local_index = LocalCellPosition::<S>::new(IVec3::new(
            position.x & mask,
            position.y & mask,
            position.z & mask,
        ))
        .index();

        Some((
            CellID::new(cached.id, local_index),
            // uninitialized chunks return C::default()
            cached.chunk.get_cell(local_index),
        ))
    }
}

// implemented on &CellView because GridExt::segment_overlap_iter takes the grid by value
impl<'v, 'g, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> Grid<'v>
    for &'v VoxelCellView<'g, C, S, T>
{
    type Cell = (CellID<S>, C);

    fn get_cell(&self, position: IVec3) -> Option<Self::Cell> {
        self.lookup(position)
    }

    fn aabb(&self) -> Aabb<IVec3> {
        Aabb::new(IVec3::new(0, 0, 0), self.grid.cell_dimensions)
    }
}

#[derive(Debug, Clone)]
pub struct BreadthFirstCellIDIter<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
> {
    grid: &'a VoxelGrid<C, S, T>,
    queue: VecDeque<CellID<S>>,
    explored: HashMap<ChunkID, BitSet<u64>>,
}

impl<'a, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>>
    BreadthFirstCellIDIter<'a, C, S, T>
{
    /// instantiates a new iterator
    /// seed cells outside the grid and duplicate seed cells are dropped
    pub fn new(grid: &'a VoxelGrid<C, S, T>, queue: VecDeque<CellID<S>>) -> Self {
        let mut explored: HashMap<ChunkID, BitSet<u64>> = HashMap::new();
        let mut seeds = VecDeque::with_capacity(queue.len());

        for cell in queue {
            if !grid.contains_cell(cell) {
                continue;
            }
            let set = explored
                .entry(cell.chunk_id())
                .or_insert_with(|| BitSet::with_capacity_general(S::SIZE));
            if set.insert(cell.local_index()) {
                seeds.push_back(cell);
            }
        }

        Self {
            grid,
            queue: seeds,
            explored,
        }
    }

    /// the length of the underlying queue
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }
}

impl<'a, C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> Iterator
    for BreadthFirstCellIDIter<'a, C, S, T>
{
    type Item = CellID<S>;

    fn next(&mut self) -> Option<Self::Item> {
        // try to emit top of queue
        let current = self.queue.pop_front()?;

        const DELTAS: [nalgebra::Vector3<isize>; 6] = [
            nalgebra::Vector3::new(-1, 0, 0),
            nalgebra::Vector3::new(1, 0, 0),
            nalgebra::Vector3::new(0, -1, 0),
            nalgebra::Vector3::new(0, 1, 0),
            nalgebra::Vector3::new(0, 0, -1),
            nalgebra::Vector3::new(0, 0, 1),
        ];

        // add unexplored neighbors to queue
        let current_pos = self.grid.get_cell_position(current).unwrap().0;
        for delta in DELTAS {
            let neighbor_pos = current_pos + delta;

            if let Some(neighbor_id) = self.grid.get_cell_id(CellPosition(neighbor_pos)) {
                let chunk_entry = self.explored.entry(neighbor_id.chunk_id());
                let chunk_explored_set =
                    chunk_entry.or_insert_with(|| BitSet::with_capacity_general(S::SIZE));

                let local_index = neighbor_id.local_index();
                if !chunk_explored_set.contains(local_index) {
                    self.queue.push_back(neighbor_id);
                    chunk_explored_set.insert(local_index);
                }
            }
        }

        Some(current)
    }
}

/// a grid of chunk ids, used to walk a segment across the chunk lattice
#[derive(Debug, Clone, Copy)]
pub struct ChunkIdGrid {
    chunk_dimensions: IVec3,
}

impl Grid<'static> for ChunkIdGrid {
    type Cell = ChunkID;

    fn get_cell(&self, position: IVec3) -> Option<ChunkID> {
        ChunkID::try_from_position_and_dimensions(ChunkPosition(position), self.chunk_dimensions)
    }

    fn aabb(&self) -> Aabb<IVec3> {
        Aabb::new(IVec3::new(0, 0, 0), self.chunk_dimensions)
    }
}

/// a grid of cell ids, used to walk a segment across the cells of a voxel grid
/// without borrowing the grid
#[derive(Debug, Clone, Copy)]
pub struct CellIdGrid<S: ChunkSize> {
    cell_dimensions: IVec3,
    chunk_dimensions: IVec3,
    marker: PhantomData<S>,
}

impl<S: ChunkSize> Grid<'static> for CellIdGrid<S> {
    type Cell = CellID<S>;

    fn get_cell(&self, position: IVec3) -> Option<CellID<S>> {
        let dims = self.cell_dimensions;
        if position.x < 0
            || position.y < 0
            || position.z < 0
            || position.x >= dims.x
            || position.y >= dims.y
            || position.z >= dims.z
        {
            return None;
        }

        let shift = S::POWER_OF_TWO;
        let mask = S::SIDE_LENGTH as isize - 1;

        let chunk_id = ChunkID::try_from_position_and_dimensions(
            ChunkPosition(IVec3::new(
                position.x >> shift,
                position.y >> shift,
                position.z >> shift,
            )),
            self.chunk_dimensions,
        )?;
        let local_index = LocalCellPosition::<S>::new(IVec3::new(
            position.x & mask,
            position.y & mask,
            position.z & mask,
        ))
        .index();

        Some(CellID::new(chunk_id, local_index))
    }

    fn aabb(&self) -> Aabb<IVec3> {
        Aabb::new(IVec3::new(0, 0, 0), self.cell_dimensions)
    }
}
