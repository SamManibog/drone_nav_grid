use std::{fmt::Debug, hash::Hash, marker::PhantomData};

use bit_set::BitSet;

use crate::{
    math::{Aabb, AabbCellIter, FVec3, Grid, GridSegmentOverlapIter, IVec3, LineSegment},
    voxel_grid::cell_identifiers::LocalCellPosition,
};

mod size_sealer {
    use super::{ChunkSize8, ChunkSize16, ChunkSize32, ChunkSize64};

    pub trait Seal {}
    impl Seal for ChunkSize8 {}
    impl Seal for ChunkSize16 {}
    impl Seal for ChunkSize32 {}
    impl Seal for ChunkSize64 {}
}

pub trait ChunkSize:
    size_sealer::Seal + 'static + PartialEq + Eq + Clone + Copy + Debug + Sized + Hash
{
    /// the power of 2 the side length is
    const POWER_OF_TWO: usize;

    /// the number of cells per side of the chunk
    const SIDE_LENGTH: usize;

    /// the number of cells in the chunk
    const SIZE: usize;

    /// the dimensions of the chunk in cell widths
    const DIMENSIONS: IVec3;

    /// the aabb of the chunk in floating-point cell width
    /// cell[0, 0, 0] is at (0, 0, 0)
    const NORMALIZED_AABB: Aabb<FVec3>;

    fn new() -> Self;
}

pub trait ChunkState<C: Copy + Default + PartialEq, S: ChunkSize>: Default + Clone {
    /// called after the chunk is initialized
    fn on_init(&mut self);

    /// called after the chunk is uninitialized
    fn on_uninit(&mut self);
}

impl<C: Copy + Default + PartialEq, S: ChunkSize> ChunkState<C, S> for () {
    fn on_init(&mut self) {}
    fn on_uninit(&mut self) {}
}

#[derive(Debug, Default, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StatelessChunkState;

impl<C: Copy + Default + PartialEq, S: ChunkSize> ChunkState<C, S> for StatelessChunkState {
    fn on_init(&mut self) {}
    fn on_uninit(&mut self) {}
}

macro_rules! ChunkSizeDef {
    ($name:ident, $pow:literal) => {
        /// A ZST marker for a chunk's size
        #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name;

        impl ChunkSize for $name {
            const POWER_OF_TWO: usize = $pow;
            const SIDE_LENGTH: usize = 1 << Self::POWER_OF_TWO;
            const SIZE: usize = Self::SIDE_LENGTH.pow(3);
            const DIMENSIONS: IVec3 = IVec3::new(
                Self::SIDE_LENGTH as isize,
                Self::SIDE_LENGTH as isize,
                Self::SIDE_LENGTH as isize,
            );
            const NORMALIZED_AABB: Aabb<FVec3> = Aabb {
                min: FVec3::new(0.0, 0.0, 0.0),
                max: FVec3::new(
                    Self::SIDE_LENGTH as f32,
                    Self::SIDE_LENGTH as f32,
                    Self::SIDE_LENGTH as f32,
                ),
            };

            fn new() -> Self {
                $name
            }
        }
    };
}

ChunkSizeDef!(ChunkSize8, 3);
ChunkSizeDef!(ChunkSize16, 4);
ChunkSizeDef!(ChunkSize32, 5);
ChunkSizeDef!(ChunkSize64, 6);

impl<S: ChunkSize> Grid<'static> for S {
    type Cell = usize;

    fn get_cell(&self, position: IVec3) -> Option<Self::Cell> {
        if position.x >= Self::DIMENSIONS.x
            || position.y >= Self::DIMENSIONS.y
            || position.z >= Self::DIMENSIONS.z
        {
            std::hint::cold_path();
            return None;
        }

        Some(LocalCellPosition::<S>::new(position).index())
    }

    fn aabb(&self) -> Aabb<IVec3> {
        Aabb {
            min: IVec3::origin(),
            max: Self::DIMENSIONS,
        }
    }
}

pub trait ChunkTrait {
    type Cell: Copy + Default + PartialEq;
    type Size: ChunkSize;

    const POWER_OF_TWO: usize = Self::Size::SIZE;
    const SIDE_LENGTH: usize = Self::Size::SIDE_LENGTH;
    const SIZE: usize = Self::Size::SIZE;
    const DIMENSIONS: IVec3 = Self::Size::DIMENSIONS;
}

impl<C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> ChunkTrait
    for Chunk<C, S, T>
{
    type Cell = C;
    type Size = S;
}

#[derive(Debug)]
pub struct Chunk<
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S> = StatelessChunkState,
> {
    /// the cells of each chunk
    cells: Vec<C>,
    marker: PhantomData<S>,

    /// user state for the chunk
    pub user_state: T,
}

impl<C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> Clone for Chunk<C, S, T> {
    fn clone(&self) -> Self {
        Self {
            cells: self.cells.clone(),
            marker: PhantomData,

            user_state: self.user_state.clone(),
        }
    }
}

impl<C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> Default for Chunk<C, S, T> {
    fn default() -> Self {
        Self {
            cells: Vec::with_capacity(0),
            marker: PhantomData,

            user_state: T::default(),
        }
    }
}

impl<C: Copy + Default + PartialEq, S: ChunkSize, T: ChunkState<C, S>> Chunk<C, S, T> {
    pub const LENGTH_POWER_OF_2: usize = S::POWER_OF_TWO;
    pub const SIDE_LENGTH: usize = S::SIDE_LENGTH;
    pub const SIZE: usize = S::SIZE;
    pub const DIMENSIONS: IVec3 = S::DIMENSIONS;

    pub fn init(&mut self) {
        if self.is_init() {
            return;
        }

        self.cells = vec![Default::default(); S::SIZE];
        self.user_state.on_init();
    }

    pub fn is_init(&self) -> bool {
        !self.cells.is_empty()
    }

    pub fn clear(&mut self) {
        if !self.is_init() {
            return;
        }

        self.cells = Vec::with_capacity(0);
        self.user_state.on_uninit();
    }

    pub fn get_cell(&self, index: usize) -> C {
        debug_assert!(index < Self::SIZE);

        self.cells.get(index).copied().unwrap_or_default()
    }

    // gets the cell mutably, panics if not yet initialized
    pub fn get_cell_mut<U>(
        &mut self,
        index: usize,
        mutator: impl FnOnce(usize, &mut C, &mut T) -> U,
    ) -> U {
        debug_assert!(index < Self::SIZE);
        debug_assert!(self.is_init());

        unsafe {
            mutator(
                index,
                self.cells.get_unchecked_mut(index),
                &mut self.user_state,
            )
        }
    }

    /// performs a depth-first traversal on a chunk starting from a given list of cells (predicate
    /// will be run on the passed cells),
    /// returning lists of the cells to check in outside chunks, identified using local cell positions.
    /// Cells for which predicate returns true are considered traversible.
    /// panics in debug builds if not yet initialized or a start index is out of bounds
    ///
    /// uses 6-neighbor (face) connectivity
    pub fn depth_first_traversal(
        &mut self,
        start_cells: impl Iterator<Item = usize>,
        mut predicate: impl FnMut(usize, &C, &T) -> bool,
        mut mutator: impl FnMut(usize, &mut C, &mut T),
    ) -> ChunkBoundaryTraversalResult<S> {
        debug_assert!(self.is_init());

        let mut traversal_result = ChunkBoundaryTraversalResult::empty();
        let mut stack = Vec::with_capacity(Self::SIZE);
        let mut explored = BitSet::<u64>::with_capacity_general(Self::SIZE);

        // traverse start nodes
        for index in start_cells {
            debug_assert!(index < Self::SIZE);
            if predicate(index, &self.cells[index], &self.user_state) {
                stack.push(index);
            }
            explored.insert(index);
        }

        while let Some(curr_index) = stack.pop() {
            mutator(
                curr_index,
                &mut self.cells[curr_index],
                &mut self.user_state,
            );

            let pos = LocalCellPosition::<S>::from_index(curr_index);
            let x = pos.x;
            let y = pos.y;
            let z = pos.z;
            let side_len = S::SIDE_LENGTH as isize;

            // +x neighbor
            if x + 1 < side_len {
                let neighbor_index = LocalCellPosition::<S>::new(IVec3::new(x + 1, y, z)).index();
                if predicate(
                    neighbor_index,
                    &self.cells[neighbor_index],
                    &self.user_state,
                ) && !explored.contains(neighbor_index)
                {
                    explored.insert(neighbor_index);
                    stack.push(neighbor_index);
                }
            } else {
                let boundary_pos = LocalCellPosition::<S>::new(IVec3::new(0, y, z));
                traversal_result.pos_x.push(boundary_pos);
            }

            // -x neighbor
            if x - 1 >= 0 {
                let neighbor_index = LocalCellPosition::<S>::new(IVec3::new(x - 1, y, z)).index();
                if predicate(
                    neighbor_index,
                    &self.cells[neighbor_index],
                    &self.user_state,
                ) && !explored.contains(neighbor_index)
                {
                    explored.insert(neighbor_index);
                    stack.push(neighbor_index);
                }
            } else {
                let boundary_pos = LocalCellPosition::<S>::new(IVec3::new(side_len - 1, y, z));
                traversal_result.neg_x.push(boundary_pos);
            }

            // +y neighbor
            if y + 1 < side_len {
                let neighbor_index = LocalCellPosition::<S>::new(IVec3::new(x, y + 1, z)).index();
                if predicate(
                    neighbor_index,
                    &self.cells[neighbor_index],
                    &self.user_state,
                ) && !explored.contains(neighbor_index)
                {
                    explored.insert(neighbor_index);
                    stack.push(neighbor_index);
                }
            } else {
                let boundary_pos = LocalCellPosition::<S>::new(IVec3::new(x, 0, z));
                traversal_result.pos_y.push(boundary_pos);
            }

            // -y neighbor
            if y - 1 >= 0 {
                let neighbor_index = LocalCellPosition::<S>::new(IVec3::new(x, y - 1, z)).index();
                if predicate(
                    neighbor_index,
                    &self.cells[neighbor_index],
                    &self.user_state,
                ) && !explored.contains(neighbor_index)
                {
                    explored.insert(neighbor_index);
                    stack.push(neighbor_index);
                }
            } else {
                let boundary_pos = LocalCellPosition::<S>::new(IVec3::new(x, side_len - 1, z));
                traversal_result.neg_y.push(boundary_pos);
            }

            // +z neighbor
            if z + 1 < side_len {
                let neighbor_index = LocalCellPosition::<S>::new(IVec3::new(x, y, z + 1)).index();
                if predicate(
                    neighbor_index,
                    &self.cells[neighbor_index],
                    &self.user_state,
                ) && !explored.contains(neighbor_index)
                {
                    explored.insert(neighbor_index);
                    stack.push(neighbor_index);
                }
            } else {
                let boundary_pos = LocalCellPosition::<S>::new(IVec3::new(x, y, 0));
                traversal_result.pos_z.push(boundary_pos);
            }

            // -z neighbor
            if z - 1 >= 0 {
                let neighbor_index = LocalCellPosition::<S>::new(IVec3::new(x, y, z - 1)).index();
                if predicate(
                    neighbor_index,
                    &self.cells[neighbor_index],
                    &self.user_state,
                ) && !explored.contains(neighbor_index)
                {
                    explored.insert(neighbor_index);
                    stack.push(neighbor_index);
                }
            } else {
                let boundary_pos = LocalCellPosition::<S>::new(IVec3::new(x, y, side_len - 1));
                traversal_result.neg_z.push(boundary_pos);
            }
        }

        traversal_result
    }

    /// returns an iterator over the cells in this chunk
    /// returns none if not initialized
    pub fn iter(&self) -> Option<ChunkIter<'_, C, S, AabbIndexIter<S>>> {
        self.iter_bounded(
            LocalCellPosition::new(IVec3::new(0, 0, 0)),
            LocalCellPosition::new(Self::DIMENSIONS),
        )
    }

    /// returns an iterator with the given end-exclusive bounds
    /// returns none if not initialized
    pub fn iter_bounded(
        &self,
        min: LocalCellPosition<S>,
        max: LocalCellPosition<S>,
    ) -> Option<ChunkIter<'_, C, S, AabbIndexIter<S>>> {
        if self.is_init() {
            Some(ChunkIter {
                cells: &self.cells,
                index_iter: AabbIndexIter::new(min, max),
                marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// returns a mutable iterator over the cells in this chunk
    /// returns none if not initialized
    pub fn iter_mut(&mut self) -> Option<ChunkIterMut<'_, C, S, T, AabbIndexIter<S>>> {
        self.iter_mut_bounded(
            LocalCellPosition::new(IVec3::new(0, 0, 0)),
            LocalCellPosition::new(Self::DIMENSIONS),
        )
    }

    /// returns a mutable iterator with the given end-exclusive bounds
    /// returns none if not initialized
    pub fn iter_mut_bounded(
        &mut self,
        min: LocalCellPosition<S>,
        max: LocalCellPosition<S>,
    ) -> Option<ChunkIterMut<'_, C, S, T, AabbIndexIter<S>>> {
        if self.is_init() {
            Some(ChunkIterMut {
                cells: &mut self.cells,
                user_state: &mut self.user_state,
                index_iter: AabbIndexIter::new(min, max),
                marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// returns an iterator over the cells intersecting with a line segment in this chunk
    pub fn iter_segment_overlap(
        &self,
        segment: LineSegment,
    ) -> Option<ChunkIter<'_, C, S, SegmentOverlapIter<S>>> {
        if self.is_init() {
            Some(ChunkIter {
                cells: &self.cells,
                index_iter: GridSegmentOverlapIter::new(S::new(), segment),
                marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// returns a mutable iterator over the cells intersecting with a line segment in this chunk
    pub fn iter_mut_segment_overlap(
        &mut self,
        segment: LineSegment,
    ) -> Option<ChunkIterMut<'_, C, S, T, SegmentOverlapIter<S>>> {
        if self.is_init() {
            Some(ChunkIterMut {
                cells: &mut self.cells,
                user_state: &mut self.user_state,
                index_iter: GridSegmentOverlapIter::new(S::new(), segment),
                marker: PhantomData,
            })
        } else {
            None
        }
    }
}

/// the result of potentially out-of-bounds traversing over cells in a chunk first traversal over a chunk
#[derive(Debug, Clone)]
pub struct ChunkBoundaryTraversalResult<S: ChunkSize> {
    pub pos_x: Vec<LocalCellPosition<S>>,
    pub neg_x: Vec<LocalCellPosition<S>>,
    pub pos_y: Vec<LocalCellPosition<S>>,
    pub neg_y: Vec<LocalCellPosition<S>>,
    pub pos_z: Vec<LocalCellPosition<S>>,
    pub neg_z: Vec<LocalCellPosition<S>>,
}

impl<S: ChunkSize> Default for ChunkBoundaryTraversalResult<S> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<S: ChunkSize> ChunkBoundaryTraversalResult<S> {
    fn empty() -> Self {
        Self {
            pos_x: Vec::new(),
            neg_x: Vec::new(),
            pos_y: Vec::new(),
            neg_y: Vec::new(),
            pos_z: Vec::new(),
            neg_z: Vec::new(),
        }
    }
}

/// an iterator over chunk indices in an aabb
#[derive(Debug, Clone)]
#[repr(transparent)]
pub struct AabbIndexIter<S: ChunkSize> {
    /// iterator over cells
    cell_iter: AabbCellIter,

    marker: PhantomData<S>,
}

pub type SegmentOverlapIter<S> = GridSegmentOverlapIter<'static, S>;

impl<S: ChunkSize> AabbIndexIter<S> {
    pub fn new(mut start: LocalCellPosition<S>, mut end: LocalCellPosition<S>) -> Self {
        start = start.entrywise_min(&LocalCellPosition::new(S::DIMENSIONS));
        end = end.entrywise_min(&LocalCellPosition::new(
            S::DIMENSIONS + IVec3::new(1, 1, 1).coords,
        ));
        Self {
            cell_iter: AabbCellIter::new(Aabb::new(start.0, end.0)),
            marker: PhantomData,
        }
    }
}

impl<S: ChunkSize> Iterator for AabbIndexIter<S> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.cell_iter
            .next()
            .map(|pos| LocalCellPosition::<S>::new(pos).index())
    }
}

/// a mutable iterator over a chunk
#[derive(Debug)]
pub struct ChunkIterMut<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = usize>,
> {
    cells: &'a mut [C],
    user_state: &'a mut T,
    index_iter: I,
    marker: PhantomData<S>,
}

impl<
    'a,
    C: Copy + Default + PartialEq,
    S: ChunkSize,
    T: ChunkState<C, S>,
    I: Iterator<Item = usize>,
> Iterator for ChunkIterMut<'a, C, S, T, I>
{
    type Item = (usize, &'a mut C, &'a mut T);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index_iter.next()?;
        let cell_ptr = std::ptr::from_mut(&mut self.cells[index]);
        let state_ptr = std::ptr::from_mut(&mut self.user_state);
        Some(unsafe {
            (
                index,
                cell_ptr.as_mut_unchecked(),
                state_ptr.as_mut_unchecked(),
            )
        })
    }
}

/// an iterator over a chunk
#[derive(Debug, Clone)]
pub struct ChunkIter<'a, C: Copy + Default + PartialEq, S: ChunkSize, I: Iterator<Item = usize>> {
    cells: &'a [C],
    index_iter: I,
    marker: PhantomData<S>,
}

impl<'a, C: Copy + Default + PartialEq, S: ChunkSize, I: Iterator<Item = usize>> Iterator
    for ChunkIter<'a, C, S, I>
{
    type Item = (usize, C);

    fn next(&mut self) -> Option<Self::Item> {
        let index = self.index_iter.next()?;
        Some((index, unsafe { *self.cells.get_unchecked(index) }))
    }
}
