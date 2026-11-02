use std::{iter::Peekable, marker::PhantomData};

use nalgebra::SimdPartialOrd;

pub type IVec3 = nalgebra::Point3<isize>;
pub type FVec3 = nalgebra::Point3<f32>;

pub trait Grid<'a> {
    type Cell;

    /// gets the cell at the given position
    fn get_cell(&self, position: IVec3) -> Option<Self::Cell>;

    /// gets the END-EXCLUSIVE aabb
    fn aabb(&self) -> Aabb<IVec3>;
}

pub trait GridExt<'a>: Grid<'a> {
    fn segment_overlap_iter(self, seg: LineSegment) -> GridSegmentOverlapIter<'a, Self>
    where
        Self: Sized,
    {
        GridSegmentOverlapIter::new(self, seg)
    }
}

pub fn floor_fvec3(fvec3: &FVec3) -> IVec3 {
    IVec3::new(
        fvec3.x.floor() as isize,
        fvec3.y.floor() as isize,
        fvec3.z.floor() as isize,
    )
}

pub fn ivec3_dist_sq(p1: &IVec3, p2: &IVec3) -> isize {
    let d_x = p1.x - p2.x;
    let d_y = p1.y - p2.y;
    let d_z = p1.z - p2.z;

    d_x * d_x + d_y * d_y + d_z * d_z
}

impl<'a, G: Grid<'a>> GridExt<'a> for G {}

#[derive(Debug, Clone, Copy)]
pub struct Aabb<V> {
    pub min: V,
    pub max: V,
}

impl<V> Aabb<V> {
    pub fn new(min: V, max: V) -> Self {
        Self { min, max }
    }

}

impl Aabb<IVec3> {
    /// gets the *end-exclusive* aabb of the intersection of two *end-exclusive* aabbs
    pub fn intersection(&self, other: &Self) -> Self {
        Self {
            min: IVec3::new(
                self.min.x.max(other.min.x),
                self.min.y.max(other.min.y),
                self.min.z.max(other.min.z),
            ),

            max: IVec3::new(
                self.max.x.min(other.max.x),
                self.max.y.min(other.max.y),
                self.max.z.min(other.max.z),
            ),
        }
    }

    /// checks if the Aabb is empty, assumining end-exclusive bounds
    pub fn is_empty(&self) -> bool {
        return self.min.x >= self.max.x || self.min.y >= self.max.y || self.min.z >= self.max.z;
    }
}

impl Aabb<FVec3> {
    pub fn from_sphere(center: FVec3, radius: f32) -> Self {
        let rad_cube = FVec3::new(radius, radius, radius).coords;
        Self {
            min: center - rad_cube,
            max: center + rad_cube,
        }
    }

    pub fn segment_intersection(&self, segment: &LineSegment) -> Option<LineSegment> {
        segment_aabb_intersection(self, segment)
    }

    /// checks if the Aabb is empty, assumining end-exclusive bounds
    pub fn is_empty(&self) -> bool {
        return self.min.x >= self.max.x || self.min.y >= self.max.y || self.min.z >= self.max.z;
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LineSegment {
    pub start: FVec3,
    pub end: FVec3,
}

impl LineSegment {
    pub fn aabb_intersection(&self, aabb: &Aabb<FVec3>) -> Option<Self> {
        segment_aabb_intersection(aabb, self)
    }
}

/// gets the distance of a point to the path
pub fn distance_to_path(point: FVec3, mut path: Peekable<impl Iterator<Item = FVec3>>) -> f32 {
    let first = match path.next() {
        Some(p) => p,
        None => return f32::INFINITY,
    };

    // If the path only contains a single point
    if path.peek().is_none() {
        return nalgebra::distance(&point, &first);
    }

    let mut min_dist_sq = f32::INFINITY;
    let mut prev = first;

    while let Some(&curr) = path.peek() {
        // Advance the iterator
        path.next();

        // Vector AB (segment) and AP (point from start of segment)
        let ab_x = curr.x - prev.x;
        let ab_y = curr.y - prev.y;
        let ab_z = curr.z - prev.z;

        let ap_x = point.x - prev.x;
        let ap_y = point.y - prev.y;
        let ap_z = point.z - prev.z;

        let ab_len_sq = ab_x * ab_x + ab_y * ab_y + ab_z * ab_z;

        // Project point onto segment, clamped to the segment bounds [0.0, 1.0]
        let t = if ab_len_sq == 0.0 {
            0.0
        } else {
            let dot = ap_x * ab_x + ap_y * ab_y + ap_z * ab_z;
            (dot / ab_len_sq).clamp(0.0, 1.0)
        };

        // Find the closest point on the segment
        let closest_x = prev.x + t * ab_x;
        let closest_y = prev.y + t * ab_y;
        let closest_z = prev.z + t * ab_z;

        let pdx = point.x - closest_x;
        let pdy = point.y - closest_y;
        let pdz = point.z - closest_z;

        let d_sq = pdx * pdx + pdy * pdy + pdz * pdz;
        if d_sq < min_dist_sq {
            min_dist_sq = d_sq;
        }

        prev = curr;
    }

    min_dist_sq.sqrt()
}

/// gets the line segment representing all points shared by the AABB and segment
pub fn segment_aabb_intersection(aabb: &Aabb<FVec3>, seg: &LineSegment) -> Option<LineSegment> {
    let d = FVec3::new(
        seg.end.x - seg.start.x,
        seg.end.y - seg.start.y,
        seg.end.z - seg.start.z,
    );

    let mut tmin = 0.0_f32;
    let mut tmax = 1.0_f32;

    let axes = [
        (seg.start.x, d.x, aabb.min.x, aabb.max.x),
        (seg.start.y, d.y, aabb.min.y, aabb.max.y),
        (seg.start.z, d.z, aabb.min.z, aabb.max.z),
    ];

    for (p, dir, box_min, box_max) in axes {
        if dir.abs() < 1e-7 {
            // Segment is parallel to this slab. If origin is outside, no intersection.
            if p < box_min || p > box_max {
                return None;
            }
        } else {
            let inv_dir = 1.0 / dir;
            let mut t1 = (box_min - p) * inv_dir;
            let mut t2 = (box_max - p) * inv_dir;

            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }

            tmin = tmin.max(t1);
            tmax = tmax.min(t2);

            if tmin > tmax {
                return None;
            }
        }
    }

    // Ensure intersection is within the [0, 1] segment bounds
    if tmin > tmax || tmax < 0.0 || tmin > 1.0 {
        return None;
    }

    let t_start = tmin.clamp(0.0, 1.0);
    let t_end = tmax.clamp(0.0, 1.0);

    let clipped_start = FVec3::new(
        seg.start.x + t_start * d.x,
        seg.start.y + t_start * d.y,
        seg.start.z + t_start * d.z,
    );

    let clipped_end = FVec3::new(
        seg.start.x + t_end * d.x,
        seg.start.y + t_end * d.y,
        seg.start.z + t_end * d.z,
    );

    Some(LineSegment {
        start: clipped_start,
        end: clipped_end,
    })
}

pub struct GridSegmentOverlapIter<'a, G: Grid<'a>> {
    grid: G,
    aabb: Aabb<IVec3>,

    pos: IVec3,
    step: IVec3,
    t_max: FVec3,
    t_delta: FVec3,
    bounds: IVec3,
    done: bool,

    marker: PhantomData<&'a ()>,
}

impl<'a, G: Grid<'a>> GridSegmentOverlapIter<'a, G> {
    /// an iterator that yields nothing
    fn finished(grid: G, aabb: Aabb<IVec3>) -> Self {
        Self {
            grid,
            aabb,
            pos: IVec3::origin(),
            step: IVec3::origin(),
            t_max: FVec3::origin(),
            t_delta: FVec3::origin(),
            bounds: IVec3::origin(),
            done: true,
            marker: PhantomData,
        }
    }

    pub fn new(grid: G, ray: LineSegment) -> Self {
        let aabb = grid.aabb();
        if aabb.is_empty() {
            return Self::finished(grid, aabb);
        }
        let faabb = Aabb {
            min: aabb.min.cast(),
            max: aabb.max.cast(),
        };

        let clipped = match segment_aabb_intersection(&faabb, &ray) {
            Some(segment) => segment,
            None => return Self::finished(grid, aabb),
        };

        let start = clipped.start;
        let end = clipped.end;

        // Correctly clamp to inclusive valid voxel range [min, max - 1]
        let max_inclusive = IVec3::new(aabb.max.x - 1, aabb.max.y - 1, aabb.max.z - 1);

        let pos = IVec3::new(
            start.x.floor() as isize,
            start.y.floor() as isize,
            start.z.floor() as isize,
        )
        .simd_clamp(aabb.min, max_inclusive);

        let end_pos = IVec3::new(
            end.x.floor() as isize,
            end.y.floor() as isize,
            end.z.floor() as isize,
        )
        .simd_clamp(aabb.min, max_inclusive);

        let dir = FVec3::new(end.x - start.x, end.y - start.y, end.z - start.z);

        let mut step = IVec3::new(0, 0, 0);
        let mut t_max = FVec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut t_delta = FVec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);

        if dir.x.abs() > 1e-7 {
            step.x = if dir.x > 0.0 { 1 } else { -1 };
            let next_boundary = if step.x > 0 {
                (pos.x + 1) as f32
            } else {
                pos.x as f32
            };
            t_max.x = (next_boundary - start.x) / dir.x;
            t_delta.x = (step.x as f32) / dir.x;
        }

        if dir.y.abs() > 1e-7 {
            step.y = if dir.y > 0.0 { 1 } else { -1 };
            let next_boundary = if step.y > 0 {
                (pos.y + 1) as f32
            } else {
                pos.y as f32
            };
            t_max.y = (next_boundary - start.y) / dir.y;
            t_delta.y = (step.y as f32) / dir.y;
        }

        if dir.z.abs() > 1e-7 {
            step.z = if dir.z > 0.0 { 1 } else { -1 };
            let next_boundary = if step.z > 0 {
                (pos.z + 1) as f32
            } else {
                pos.z as f32
            };
            t_max.z = (next_boundary - start.z) / dir.z;
            t_delta.z = (step.z as f32) / dir.z;
        }

        Self {
            grid,

            aabb,
            pos,
            step,
            t_max,
            t_delta,
            bounds: end_pos,
            done: false,

            marker: PhantomData,
        }
    }
}

impl<'a, G: Grid<'a>> Iterator for GridSegmentOverlapIter<'a, G> {
    type Item = G::Cell;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        let cell = match self.grid.get_cell(self.pos) {
            Some(c) => c,
            None => {
                self.done = true;
                return None;
            }
        };

        if self.pos == self.bounds {
            self.done = true;
            return Some(cell);
        }

        if self.t_max.x.min(self.t_max.y).min(self.t_max.z) > 1.0 {
            self.done = true;
            return Some(cell);
        }

        // Step along the axis with the minimum t_max
        if self.t_max.x < self.t_max.y && self.t_max.x < self.t_max.z {
            self.pos.x += self.step.x;
            self.t_max.x += self.t_delta.x;
        } else if self.t_max.y < self.t_max.z {
            self.pos.y += self.step.y;
            self.t_max.y += self.t_delta.y;
        } else {
            self.pos.z += self.step.z;
            self.t_max.z += self.t_delta.z;
        }

        // Correct boundary check against grid min/max bounds (supports non-zero origin grids)
        if self.pos.x < self.aabb.min.x
            || self.pos.x >= self.aabb.max.x
            || self.pos.y < self.aabb.min.y
            || self.pos.y >= self.aabb.max.y
            || self.pos.z < self.aabb.min.z
            || self.pos.z >= self.aabb.max.z
        {
            self.done = true;
        }

        Some(cell)
    }
}

pub trait IDistanceHeuristic {
    fn apply(p1: IVec3, p2: IVec3) -> u64;
}

#[derive(Debug, Clone, Copy)]
pub struct IManhattanDistance;

impl IDistanceHeuristic for IManhattanDistance {
    fn apply(p1: IVec3, p2: IVec3) -> u64 {
        p1.x.abs_diff(p2.x) as u64 + p1.y.abs_diff(p2.y) as u64 + p1.z.abs_diff(p2.z) as u64
    }
}

#[derive(Debug, Clone, Copy)]
pub struct IDistance;

impl IDistanceHeuristic for IDistance {
    fn apply(p1: IVec3, p2: IVec3) -> u64 {
        let x_dist = p1.x.abs_diff(p2.x) as u64;
        let y_dist = p1.y.abs_diff(p2.y) as u64;
        let z_dist = p1.z.abs_diff(p2.z) as u64;
        x_dist * x_dist + y_dist * y_dist + z_dist * z_dist
    }
}

pub trait FDistanceHeuristic {
    fn apply(p1: FVec3, p2: FVec3) -> f32;
}

#[derive(Debug, Clone)]
pub struct AabbCellIter {
    /// next cell position to return state of
    next: IVec3,

    /// first cell position to look at, inclusive
    start: IVec3,

    /// last cell position to look at, exclusive
    end: IVec3,
}

impl AabbCellIter {
    /// creates a new iterator
    /// cells iterated over are end-exclusive
    pub fn new(aabb: Aabb<IVec3>) -> Self {
        let start = aabb.min;
        let end = aabb.max;
        if start.x >= end.x || start.y >= end.y || start.z >= end.z {
            return Self {
                next: end,
                start: end,
                end,
            };
        }

        let start = IVec3::new(
            start.x.min(end.x - 1),
            start.y.min(end.y - 1),
            start.z.min(end.z - 1),
        );

        Self {
            next: start,
            start,
            end,
        }
    }

    /// gets the start coordinate,
    /// which is the min bound
    pub const fn start(&self) -> IVec3 {
        self.start
    }

    /// gets the end-exclusive max coordinate
    pub const fn end(&self) -> IVec3 {
        self.end
    }
}

impl Iterator for AabbCellIter {
    type Item = IVec3;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next.z >= self.end.z {
            return None;
        }

        let out = self.next;

        self.next.x += 1;
        if self.next.x >= self.end.x {
            self.next.x = self.start.x;
            self.next.y += 1;
        }

        if self.next.y >= self.end.y {
            self.next.y = self.start.y;
            self.next.z += 1;
        }

        Some(out)
    }
}
