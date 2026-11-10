use std::{cmp::Ordering, collections::BinaryHeap};

use bit_set::BitSet;
use bit_vec::BitVec;

use crate::{heap_entries::MaxHeapEntry, math::IVec2};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

impl Corner {
    /// gets the opposite corner
    pub fn opposite(self) -> Self {
        match self {
            Corner::TopLeft => Corner::BottomRight,
            Corner::TopRight => Corner::BottomLeft,
            Corner::BottomLeft => Corner::TopRight,
            Corner::BottomRight => Corner::TopLeft,
        }
    }

    /// gets the pair of nearest corners between the two cells
    /// +x is right
    /// -x is left
    /// +y is down
    /// -y is up
    ///
    /// returns (corner of cell1 nearest to cell2, corner of cell2 nearest to cell1)
    pub fn nearest_corners(cell1: IVec2, cell2: IVec2) -> (Corner, Corner) {
        match (cell1.x.cmp(&cell2.x), cell1.y.cmp(&cell2.y)) {
            (Ordering::Equal, Ordering::Equal) => (Corner::TopLeft, Corner::TopLeft),

            (Ordering::Equal, Ordering::Less) => (Corner::BottomLeft, Corner::TopLeft),
            (Ordering::Equal, Ordering::Greater) => (Corner::TopLeft, Corner::BottomLeft),

            (Ordering::Less, Ordering::Equal) => (Corner::TopRight, Corner::TopLeft),
            (Ordering::Greater, Ordering::Equal) => (Corner::TopLeft, Corner::TopRight),

            (Ordering::Less, Ordering::Less) => (Corner::BottomRight, Corner::TopLeft),
            (Ordering::Less, Ordering::Greater) => (Corner::TopRight, Corner::BottomLeft),
            (Ordering::Greater, Ordering::Less) => (Corner::BottomLeft, Corner::TopRight),
            (Ordering::Greater, Ordering::Greater) => (Corner::TopLeft, Corner::BottomRight),
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WaypointMoveCmd {
    /// repeat command until cant anymore
    Inf(IVec2),

    /// repeat command once
    One(IVec2),
}

// movement pattern for top left to bottom right corner
// assumes positive y is down, positive x is right
const MOVEMENT_PATTERN: [WaypointMoveCmd; 6] = [
    WaypointMoveCmd::Inf(IVec2::new(0, 1)),
    WaypointMoveCmd::One(IVec2::new(1, 0)),
    WaypointMoveCmd::Inf(IVec2::new(0, -1)),
    WaypointMoveCmd::Inf(IVec2::new(1, 0)),
    WaypointMoveCmd::One(IVec2::new(0, 1)),
    WaypointMoveCmd::Inf(IVec2::new(-1, 0)),
];

#[derive(Debug, Clone)]
pub struct ScoreGrid {
    dimensions: IVec2,
    scores: Vec<f32>,
}

fn grid_index(pos: IVec2, x_dimension: isize) -> usize {
    (pos.x + x_dimension * pos.y) as usize
}

fn grid_pos(index: usize, x_dimension: isize) -> IVec2 {
    IVec2::new(
        (index as isize).rem_euclid(x_dimension),
        index as isize / x_dimension,
    )
}

fn pos_dist(pos1: IVec2, pos2: IVec2) -> f32 {
    let rel_pos = pos1 - pos2.coords;
    f32::sqrt((rel_pos.x * rel_pos.x + rel_pos.y * rel_pos.y) as f32)
}

impl ScoreGrid {
    pub fn new(score_matrix: Vec<Vec<f32>>) -> Result<Self, String> {
        if score_matrix.is_empty() {
            return Err("Score matrix cannot be empty".to_string());
        }
        let width = score_matrix[0].len();
        if width == 0 {
            return Err("Score matrix rows cannot be empty".to_string());
        }
        let height = score_matrix.len();

        let mut scores = Vec::with_capacity(width * height);
        for (y, row) in score_matrix.into_iter().enumerate() {
            if row.len() != width {
                return Err(format!(
                    "Row {} has length {}, expected {}",
                    y,
                    row.len(),
                    width
                ));
            }
            for val in row {
                if !val.is_finite() {
                    return Err("Score values must be finite".to_string());
                }
                scores.push(val);
            }
        }

        Ok(Self {
            dimensions: IVec2::new(width as isize, height as isize),
            scores,
        })
    }

    pub fn map_indices(&self, indices: impl Iterator<Item = usize>) -> impl Iterator<Item = IVec2> {
        indices.map(|i| self.node_pos(i))
    }

    pub fn node_index(&self, pos: IVec2) -> usize {
        grid_index(pos, self.dimensions.x)
    }

    pub fn contains_pos(&self, pos: IVec2) -> bool {
        pos.x >= 0 && pos.x < self.dimensions.x && pos.y >= 0 && pos.y < self.dimensions.y
    }

    pub fn node_pos(&self, index: usize) -> IVec2 {
        grid_pos(index, self.dimensions.x)
    }

    pub fn get_score(&self, pos: IVec2) -> f32 {
        self.scores[self.node_index(pos)]
    }

    fn get_score_by_index(&self, index: usize) -> f32 {
        self.scores[index]
    }

    pub fn node_count(&self) -> usize {
        (self.dimensions.x * self.dimensions.y) as usize
    }

    pub fn dimensions(&self) -> IVec2 {
        self.dimensions
    }
}

/// generates a snaking path through a grid with the given dimensions
pub fn snaking_path(dimensions: IVec2) -> Vec<usize> {
    let mut path = Vec::new();
    for i in 0..dimensions.x {
        if i % 2 == 0 {
            for j in 0..dimensions.y {
                path.push(grid_index(IVec2::new(i, j), dimensions.x))
            }
        } else {
            for j in (0..dimensions.y).into_iter().rev() {
                path.push(grid_index(IVec2::new(i, j), dimensions.x))
            }
        }
    }
    path
}

#[derive(Debug, Clone)]
pub struct PathPlanningProblem {
    /// the starting node (need not be explored as first move)
    pub start_position: IVec2,

    /// the score of each node (want paths that maximize score)
    pub scores: ScoreGrid,

    /// the cost of visiting a node
    pub node_cost: f32,

    /// distance_multiplier * distance between nodes is the cost of traversing an edge betweeen nodes
    pub distance_multiplier: f32,

    /// the maximum cost that can be accumulated before the path ends
    pub max_cost: f32,
}

impl PathPlanningProblem {
    /// modifies the given path for validity and returns its new score
    ///
    /// DOES NOT PERFORM DUPLICATE DETECTION/REMOVAL
    ///
    /// current method for shrinking is to iteratively select and remove the lowest-scoring included
    /// nodes
    ///
    /// current method for expansion is running greedy distance/prob algorithm
    pub fn validate_and_score(&self, path: &mut Vec<usize>) -> f32 {
        todo!()
    }

    /// ensures the problem definition is valid
    pub fn validate(&self) -> Result<(), String> {
        if self.max_cost < 0.0 && !self.max_cost.is_nan() {
            return Err("Max cost must be non-negative".to_string());
        }

        if self.node_cost < 0.0 || !self.node_cost.is_finite() {
            return Err("Node cost must be non-negative and finite".to_string());
        }

        if self.distance_multiplier < 0.0 || !self.distance_multiplier.is_finite() {
            return Err("Distance multiplier must be non-negative and finite".to_string());
        }

        if !self.scores.contains_pos(self.start_position) {
            return Err("Start position is outside grid dimensions".to_string());
        }

        Ok(())
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// the next chosen cell is the with the greatest score that doesnt exceed cost
    ///
    /// O(nlogn)
    pub fn greedy_score_soln(&self) -> Vec<usize> {
        let mut path = Vec::new();

        let mut heap = BinaryHeap::from_iter(
            (0..self.scores.node_count())
                .into_iter()
                .map(|i| MaxHeapEntry(self.scores.get_score_by_index(i), i)),
        );

        let mut remaining_cost = self.max_cost;

        let mut current_pos = self.start_position;

        while let Some(MaxHeapEntry(_, index)) = heap.pop()
            && remaining_cost > 0.0
        {
            let i_pos = self.scores.node_pos(index);
            let distance = pos_dist(current_pos, i_pos);
            let edge_cost = distance * self.distance_multiplier;
            if edge_cost + self.node_cost >= remaining_cost {
                continue;
            }

            path.push(index);

            current_pos = i_pos;
            remaining_cost -= edge_cost + self.node_cost;
        }

        path
    }

    /// shrinks the given path to a valid cost by removing lowest-scoring nodes first
    fn greedy_score_path_shrink(&self, path: &mut Vec<usize>) {
        todo!()
    }

    /// expands the given path based on score/distance greedy algorithm
    /// assumes the path is non-empty
    ///
    /// returns the total score of the path
    fn greedy_score_distance_ratio_path_expand(&self, path: &mut Vec<usize>) -> f32 {
        let (mut total_score, path_cost) = self.path_score_and_cost(path);
        let mut remaining_cost = self.max_cost - path_cost;
        let mut current_pos = self.scores.node_pos(path.pop().unwrap());

        // construct bitset
        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));
        for i in path.iter() {
            unexplored.insert(*i);
        }

        while remaining_cost > 0.0 {
            if self.greedy_score_distance_ratio_soln_helper(
                &mut current_pos,
                &mut unexplored,
                &mut remaining_cost,
                &mut total_score,
                path,
            ) {
                return total_score;
            }
        }

        total_score
    }

    /// returns the path's score and cost in that order (score, cost)
    pub fn path_score_and_cost(&self, path: &[usize]) -> (f32, f32) {
        let mut score = 0.0;

        let node_cost = self.node_cost * path.len() as f32;
        let mut edge_cost = 0.0;

        let mut v_iter = path.iter();
        let mut prev_pos = if let Some(prev) = v_iter.next() {
            score += self.scores.get_score_by_index(*prev);
            self.scores.node_pos(*prev)
        } else {
            return (score, node_cost);
        };

        for curr in v_iter {
            score += self.scores.get_score_by_index(*curr);

            let curr_pos = self.scores.node_pos(*curr);
            let dist = pos_dist(curr_pos, prev_pos);

            edge_cost += dist * self.distance_multiplier;

            prev_pos = curr_pos;
        }

        return (score, node_cost + edge_cost);
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// the next chosen cell is the one that maximizes (score/distance)
    ///
    /// returns (path, score)
    ///
    /// O(n^2)
    pub fn greedy_score_distance_ratio_soln(&self) -> (Vec<usize>, f32) {
        let mut path = Vec::new();

        let mut remaining_cost = self.max_cost;
        let mut current_pos: IVec2 = self.start_position;
        let mut total_score: f32 = 0.0;

        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));

        while remaining_cost > 0.0 {
            if self.greedy_score_distance_ratio_soln_helper(
                &mut current_pos,
                &mut unexplored,
                &mut remaining_cost,
                &mut total_score,
                &mut path,
            ) {
                return (path, total_score);
            }
        }

        (path, total_score)
    }

    /// runs an iteration of greedy_score_distance_ratio_soln,
    /// returns true if the current path should be returned
    fn greedy_score_distance_ratio_soln_helper(
        &self,
        current_pos: &mut IVec2,
        unexplored: &mut BitSet<u64>,
        remaining_cost: &mut f32,
        total_score: &mut f32,
        path: &mut Vec<usize>,
    ) -> bool {
        let mut max_pos = *current_pos;
        let mut max_sd_ratio = f32::NEG_INFINITY;
        let mut max_edge_cost = 0.0;
        let mut max_score = 0.0;

        'unex_loop: for i in unexplored.iter() {
            let i_pos = self.scores.node_pos(i);
            let distance = pos_dist(*current_pos, i_pos);
            let edge_cost = distance * self.distance_multiplier;
            if edge_cost + self.node_cost >= *remaining_cost {
                continue 'unex_loop;
            }

            let score = self.scores.get_score_by_index(i);
            let sd_ratio = if distance == 0.0 {
                score * 1000.0
            } else {
                score / distance
            };

            if sd_ratio > max_sd_ratio {
                max_sd_ratio = sd_ratio;
                max_pos = i_pos;
                max_edge_cost = edge_cost;
                max_score = max_score;
            }
        }

        // remaining nodes exhausted or unable to navigate to next node in time
        if !max_sd_ratio.is_finite() {
            return true;
        }

        // remaining nodes don't improve score
        if max_sd_ratio <= 0.0 {
            return true;
        }

        let max_idx = self.scores.node_index(max_pos);
        *current_pos = max_pos;
        unexplored.remove(max_idx);
        path.push(max_idx);
        *total_score += max_score;

        *remaining_cost -= max_edge_cost + self.node_cost;

        false
    }

    /// plans an efficient exploration path between grid cells
    ///
    /// algorithm overview:
    ///     - initial solutions to speed up convergence time:
    ///         - greedy algorithm that selects the highest probability unvisited node first
    ///         - greedy algorithm that selects unvisited node with greatest (probability/distance to node)
    ///     - end condition/output: a list of nodes representing visited
    ///     - the fitness function: sum of visited node scores
    ///     - crossover: ox1 crossover followed by greedy shrinking and expansion
    ///     - mutations:
    ///         - swap order of visitation of random chunks
    ///         - change chunk to nearby, unvisited chunk
    ///     - selection after crossover and mutation:
    ///         - certain number of top-scoring w or w/o crossover and/or mutation
    ///         - certain amount of top scoring with crossover and mutation
    pub fn genetic_soln(&self) -> Vec<usize> {
        const PASS_SIZE: usize = 8;

        // trivial case
        if self.max_cost <= 0.0 {
            return Vec::new();
        }

        todo!("complete genetic pathfinding in sprint 3")
    }

    /// performs raw crossover
    fn genetic_ox1(parent1: &[usize], parent2: &[usize]) -> Vec<usize> {
        todo!()
    }

    /// performs re-validated crossover
    fn validated_genetic_ox1(&self, parent1: &[usize], parent2: &[usize]) -> (Vec<usize>, f32) {
        let mut child = Self::genetic_ox1(parent1, parent2);
        let score = self.validate_and_score(&mut child);
        (child, score)
    }
}
