use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
};

use crate::{
    math::{FVec3, IVec3},
    voxel_grid::chunks::ChunkSize,
};

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ChunkID(usize);

impl ChunkID {
    pub const NULL: Self = Self(usize::MAX);

    pub(super) const fn new(index: usize) -> Self {
        Self(index)
    }

    pub(super) const fn index(self) -> usize {
        self.0
    }

    pub fn is_null(self) -> bool {
        self == Self::NULL
    }

    pub fn try_from_position_and_dimensions(pos: ChunkPosition, dimensions: IVec3) -> Option<Self> {
        assert!(dimensions.x >= 0 && dimensions.y >= 0 && dimensions.z >= 0);

        if pos.x >= dimensions.x
            || pos.y >= dimensions.y
            || pos.z >= dimensions.z
            || pos.x < 0
            || pos.y < 0
            || pos.z < 0
        {
            return None;
        }

        let chunk_index = (pos.x + dimensions.x * (pos.y + dimensions.y * pos.z)) as usize;
        Some(ChunkID::new(chunk_index))
    }
}

#[repr(transparent)]
#[derive(Debug, Hash)]
pub struct CellID<S: ChunkSize>(usize, PhantomData<S>);

impl<S: ChunkSize> Clone for CellID<S> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<S: ChunkSize> Copy for CellID<S> {}

impl<S: ChunkSize> PartialEq for CellID<S> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }

    fn ne(&self, other: &Self) -> bool {
        self.0 != other.0
    }
}
impl<S: ChunkSize> Eq for CellID<S> {}

impl<S: ChunkSize> PartialOrd for CellID<S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }

    fn lt(&self, other: &Self) -> bool {
        self.0.lt(&other.0)
    }

    fn gt(&self, other: &Self) -> bool {
        self.0.gt(&other.0)
    }

    fn le(&self, other: &Self) -> bool {
        self.0.le(&other.0)
    }

    fn ge(&self, other: &Self) -> bool {
        self.0.ge(&other.0)
    }
}
impl<S: ChunkSize> Ord for CellID<S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<S: ChunkSize> CellID<S> {
    pub const NULL: Self = Self(usize::MAX, PhantomData);

    pub fn is_null(self) -> bool {
        self == Self::NULL
    }

    pub(super) const fn new(chunk_id: ChunkID, local_index: usize) -> Self {
        Self((chunk_id.index() * S::SIZE) + local_index, PhantomData)
    }

    pub(super) const fn chunk_id(self) -> ChunkID {
        ChunkID::new((self.0 / S::SIZE) as usize)
    }

    pub(super) const fn global_index(self) -> usize {
        self.0
    }

    pub(super) const fn local_index(self) -> usize {
        self.0 % S::SIZE
    }

    pub(super) const fn local_position(self) -> LocalCellPosition<S> {
        LocalCellPosition::from_index(self.local_index())
    }
}

impl<S: ChunkSize> From<Option<Self>> for CellID<S> {
    fn from(value: Option<Self>) -> Self {
        match value {
            Some(id) => id,
            None => Self::NULL,
        }
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellPosition(pub IVec3);

impl Deref for CellPosition {
    type Target = IVec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for CellPosition {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl CellPosition {
    pub fn local_position<S: ChunkSize>(self) -> LocalCellPosition<S> {
        LocalCellPosition::new(IVec3::new(
            self.0.x.rem_euclid(S::SIDE_LENGTH as isize),
            self.0.y.rem_euclid(S::SIDE_LENGTH as isize),
            self.0.z.rem_euclid(S::SIDE_LENGTH as isize),
        ))
    }

    pub fn center(self) -> FVec3 {
        FVec3::new(
            self.x as f32 + 0.5,
            self.y as f32 + 0.5,
            self.z as f32 + 0.5,
        )
    }
}

#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct LocalCellPosition<S: ChunkSize>(pub IVec3, PhantomData<S>);

impl<S: ChunkSize> Clone for LocalCellPosition<S> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}
impl<S: ChunkSize> Copy for LocalCellPosition<S> {}

impl<S: ChunkSize> LocalCellPosition<S> {
    pub const fn new(vec: IVec3) -> Self {
        LocalCellPosition(vec, PhantomData)
    }

    pub fn index(self) -> usize {
        (self.x + S::SIDE_LENGTH as isize * (self.y + S::SIDE_LENGTH as isize * self.z)) as usize
    }

    pub fn try_from_fvec3(vec: FVec3) -> Option<Self> {
        let ivec = IVec3::new(
            vec.x.floor() as isize,
            vec.y.floor() as isize,
            vec.z.floor() as isize,
        );

        if ivec.x < 0
            || ivec.y < 0
            || ivec.z < 0
            || ivec.x >= S::SIDE_LENGTH as isize
            || ivec.y >= S::SIDE_LENGTH as isize
            || ivec.z >= S::SIDE_LENGTH as isize
        {
            None
        } else {
            Some(Self(ivec, PhantomData))
        }
    }

    pub const fn from_index(local_index: usize) -> Self {
        let chunk_pos_x = local_index % S::SIDE_LENGTH;
        let temp_yz = local_index / S::SIDE_LENGTH;

        let chunk_pos_y = temp_yz % S::SIDE_LENGTH;
        let chunk_pos_z = temp_yz / S::SIDE_LENGTH;

        Self::new(IVec3::new(
            chunk_pos_x as isize,
            chunk_pos_y as isize,
            chunk_pos_z as isize,
        ))
    }
}

impl<S: ChunkSize> Deref for LocalCellPosition<S> {
    type Target = IVec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<S: ChunkSize> DerefMut for LocalCellPosition<S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<S: ChunkSize> Into<usize> for LocalCellPosition<S> {
    fn into(self) -> usize {
        self.index()
    }
}

#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChunkPosition(pub IVec3);

impl Deref for ChunkPosition {
    type Target = IVec3;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for ChunkPosition {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl ChunkPosition {
    pub fn try_from_id_and_dimension(id: ChunkID, dimensions: IVec3) -> Option<Self> {
        assert!(dimensions.x >= 0 && dimensions.y >= 0 && dimensions.z >= 0);

        if id.index() >= (dimensions.x * dimensions.y * dimensions.z) as usize {
            return None;
        }

        let chunk_index = id.index();
        let chunk_pos_x = chunk_index % dimensions.x as usize;
        let temp_yz = chunk_index / dimensions.x as usize;

        let chunk_pos_y = temp_yz % dimensions.y as usize;
        let chunk_pos_z = temp_yz / dimensions.y as usize;

        Some(ChunkPosition(IVec3::new(
            chunk_pos_x as isize,
            chunk_pos_y as isize,
            chunk_pos_z as isize,
        )))
    }
}
