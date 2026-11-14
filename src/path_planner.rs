use std::{
    cmp::Ordering,
    collections::BinaryHeap,
    ops::Range,
    time::{Duration, Instant},
};

use crate::{
    heap_entries::{MaxHeapEntry, MinHeapEntry},
    math::IVec2,
};
use bit_set::BitSet;
use bit_vec::BitVec;
use priority_queue::PriorityQueue;
use rand::{Rng, RngExt, seq::SliceRandom};

#[derive(Debug, Clone)]
pub struct ScoreGrid {
    dimensions: IVec2,
    scores: Vec<f32>,
    total_score: f32,
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

fn grid_dist(pos1: IVec2, pos2: IVec2) -> f32 {
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

        let mut total_score = 0.0;

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
                total_score += val;
                scores.push(val);
            }
        }

        Ok(Self {
            dimensions: IVec2::new(width as isize, height as isize),
            scores,
            total_score,
        })
    }

    pub fn total_score(&self) -> f32 {
        self.total_score
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

    /// returns the score at the given position
    ///
    /// # Panics
    /// panics with a descriptive message if `pos` is outside the grid
    /// (a negative coordinate would otherwise wrap into a huge index)
    pub fn get_score(&self, pos: IVec2) -> f32 {
        assert!(self.contains_pos(pos), "position is outside the score grid");
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
            for j in (0..dimensions.y).rev() {
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
    /// the number of genetic iterations to run before recalculating scores and costs
    const UNDRIFT_GENETIC_ITERATIONS: usize = 10;

    /// the initial population size as a proportion of maximum possible gene count
    const POPULATION_SIZE_MULTIPLIER: f32 = 8.0;

    /// the smallest population the genetic algorithm will run with. guarantees the
    /// elite/performance/diversity bookkeeping always has individuals to work with
    const MIN_POPULATION_SIZE: usize = 16;

    /// the ratio of completely preserved individuals to total population size per generation
    /// elite individuals are not mutated, but may be selected for crossover
    const ELITE_FRACTION: f32 = 0.025;

    /// the ratio of top-performing individuals selected to pass on to the next generation
    const PERFORMANCE_FRACTION: f32 = 0.4 - Self::ELITE_FRACTION;

    /// the ratio of non-top-performing individuals selected randomly to pass on to the next generation
    /// used to diversify the gene pool
    const DIVERSITY_FRACTION: f32 = 0.1;

    /// the probability any given gene is mutated
    const MUTATION_PROBABILITY: f32 = 0.01;

    /// the probability of a crossover
    const CROSSOVER_PROBABILITY: f32 = 0.75;

    /// the maximum number of Dinkelbach iterations used to find the max-efficiency substring
    /// (converges in a handful of iterations in practice)
    const MAX_EFFICIENCY_ITERATIONS: usize = 16;

    const PASS_FRACTION: f32 =
        Self::ELITE_FRACTION + Self::PERFORMANCE_FRACTION + Self::DIVERSITY_FRACTION;

    /// evaluated at module level (see the `const _` items below the impl block);
    /// associated consts are otherwise never evaluated unless referenced
    pub const PASS_SIZE_ASSERTION: () = assert!(Self::PASS_FRACTION < 1.0_f32);
    pub const ELITE_FRACTION_ASSERTION: () =
        assert!(Self::ELITE_FRACTION <= Self::PERFORMANCE_FRACTION);

    const NUDGE_MUTATION_DELTAS: &[IVec2] = &[
        IVec2::new(2, -1),
        IVec2::new(-2, 1),
        IVec2::new(-1, 1),
        IVec2::new(0, 2),
        IVec2::new(-1, -1),
        IVec2::new(2, 0),
        IVec2::new(1, 1),
        IVec2::new(2, -2),
        IVec2::new(2, 2),
        IVec2::new(-1, -2),
        IVec2::new(-1, 0),
        IVec2::new(0, -2),
        IVec2::new(-2, 2),
        IVec2::new(0, -1),
        IVec2::new(1, -1),
        IVec2::new(-2, -1),
        IVec2::new(-2, 0),
        IVec2::new(1, 2),
        IVec2::new(2, 1),
        IVec2::new(1, 0),
        IVec2::new(-2, -2),
        IVec2::new(1, -2),
        IVec2::new(0, 1),
        IVec2::new(-1, 2),
    ];

    /// ensures the problem definition is valid
    pub fn validate(&self) -> Result<(), String> {
        // `!(x >= 0.0)` also rejects NaN
        if !(self.max_cost >= 0.0) || !self.max_cost.is_finite() {
            return Err("Max cost must be non-negative and finite".to_string());
        }

        if !(self.node_cost >= 0.0) || !self.node_cost.is_finite() {
            return Err("Node cost must be non-negative and finite".to_string());
        }

        if !(self.distance_multiplier >= 0.0) || !self.distance_multiplier.is_finite() {
            return Err("Distance multiplier must be non-negative and finite".to_string());
        }

        if !self.scores.contains_pos(self.start_position) {
            return Err("Start position is outside grid dimensions".to_string());
        }

        Ok(())
    }

    /// the cost saved by removing node `i` from the path described by `adj`, including the node
    /// cost. handles head (no prev), tail (no next), and a lone node (neither).
    ///
    /// s(i) = score of node i
    /// c(i, j) = cost of edge between i and j (distance scaled by the distance multiplier)
    /// r = root/start position
    /// C = node cost
    ///
    /// let path p = x, ..., y, ..., z
    ///   head:  c(r, x) + c(x, x + 1) - c(r, x + 1) + C
    ///   inner: c(y - 1, y) + c(y, y + 1) - c(y - 1, y + 1) + C
    ///   tail:  c(z - 1, z) + C
    fn removal_savings(&self, adj: &LinkedPath, i: usize) -> f32 {
        let (prev_opt, next_opt) = adj.neighbours(i);
        let pos = self.scores.node_pos(i);
        let prev_pos = if let Some(prev) = prev_opt {
            self.scores.node_pos(prev)
        } else {
            self.start_position
        };

        let dist = if let Some(next) = next_opt {
            let next_pos = self.scores.node_pos(next);
            grid_dist(prev_pos, pos) + grid_dist(pos, next_pos) - grid_dist(prev_pos, next_pos)
        } else {
            grid_dist(prev_pos, pos)
        };

        dist * self.distance_multiplier + self.node_cost
    }

    /// smart deduplication step for a single duplicated node: keeps only the occurrence whose
    /// removal would save the least edge cost, and adjusts score and cost accordingly
    ///
    /// the incremental cost update treats each removed occurrence independently, which is
    /// inexact when removed occurrences are adjacent; `smart_dedup` recomputes score and cost
    /// afterwards to correct for that
    ///
    /// O(n)
    fn smart_dedup_node_helper(&self, soln: &mut SolutionInfo, node_index: usize) {
        // Record path indices of duplicates
        let mut path_indices = Vec::new();
        for (i, node) in soln.path.iter().copied().enumerate() {
            if node == node_index {
                path_indices.push(i);
            }
        }

        // edge cost (in cost units, i.e. scaled by the distance multiplier) saved by removing
        // the occurrence at path index `idx`
        let comparative_path_idx_savings = |idx: usize| -> f32 {
            let pos = self.scores.node_pos(soln.path[idx]);

            let prev_pos = if idx == 0 {
                self.start_position
            } else {
                self.scores.node_pos(soln.path[idx - 1])
            };

            let dist = if idx == soln.path.len() - 1 {
                grid_dist(prev_pos, pos)
            } else {
                let next_pos = self.scores.node_pos(soln.path[idx + 1]);
                grid_dist(prev_pos, pos) + grid_dist(pos, next_pos) - grid_dist(prev_pos, next_pos)
            };

            dist * self.distance_multiplier
        };

        // Find the occurrence with the least savings
        let mut min_savings = comparative_path_idx_savings(path_indices[0]);
        let mut min_path_index = path_indices[0];

        let mut savings_cache = Vec::with_capacity(path_indices.len());
        savings_cache.push(min_savings);

        for path_index in path_indices.iter().copied().skip(1) {
            let savings = comparative_path_idx_savings(path_index);
            savings_cache.push(savings);

            if savings < min_savings {
                min_savings = savings;
                min_path_index = path_index;
            }
        }

        // Find which position in `path_indices` corresponds to our `min_path_index`
        let min_i = path_indices
            .iter()
            .position(|&idx| idx == min_path_index)
            .unwrap();

        // Adjust for removed node cost and score
        let node_score = self.scores.get_score_by_index(node_index);
        let removed_count = (path_indices.len() - 1) as f32;
        soln.score -= removed_count * node_score;
        soln.cost -= removed_count * self.node_cost;

        // Remove duplicates, adjust for edge cost/savings using occurrence index `i`
        let mut i = 0;
        soln.path.retain(|n_index| {
            if *n_index == node_index {
                let keep = i == min_i;
                if !keep {
                    soln.cost -= savings_cache[i];
                }
                i += 1;
                keep
            } else {
                true
            }
        });
    }

    /// smart deduplication step for a solution with duplicate nodes
    ///
    /// score and cost are recomputed from scratch afterwards, since the per-node incremental
    /// updates are inexact when removed duplicates are adjacent to each other
    ///
    /// O(n^2) in the worst case
    fn smart_dedup(&self, soln: &mut SolutionInfo) {
        // construct list of duplicate nodes
        let mut dupes = Vec::new();
        {
            let mut found = BitSet::<u64>::with_capacity_general(self.scores.node_count());
            let mut pushed = BitSet::<u64>::with_capacity_general(self.scores.node_count());
            for index in soln.path.iter().copied() {
                if found.contains(index) {
                    if !pushed.contains(index) {
                        dupes.push(index);
                        pushed.insert(index);
                    }
                } else {
                    found.insert(index);
                }
            }
        };

        if dupes.is_empty() {
            return;
        }

        // smart dedup on specific node
        for dupe_node in dupes {
            self.smart_dedup_node_helper(soln, dupe_node);
        }

        self.update_score_and_cost(soln);
    }

    /// shrinks a solution until it is valid, prioritizing the removal of inefficient edges
    ///
    /// the solution's path must not contain duplicate nodes (see `smart_dedup`) and its
    /// score and cost must be correct
    ///
    /// O(nlogn)
    fn greedy_efficiency_path_shrink(&self, soln: &mut SolutionInfo) {
        let mut path = LinkedPath::from_slice(&soln.path, self.scores.node_count());

        // efficiency = score created per unit of cost that removal would save.
        // The queue is a max-heap, so negate: the least efficient node pops first.
        // Removals that save nothing are treated as infinitely efficient (never
        // chosen early).
        let priority = |adj: &LinkedPath, i: usize| -> FPriority {
            let saved = self.removal_savings(adj, i);
            let efficiency = if saved <= 0.0 {
                f32::INFINITY
            } else {
                self.scores.get_score_by_index(i) / saved
            };
            FPriority(-efficiency)
        };

        // push nodes to priority queue (least efficient nodes are removed first)
        let mut queue = PriorityQueue::with_capacity(path.len());
        for node in path.iter() {
            queue.push(node, priority(&path, node));
        }

        // shrink until solution has valid cost or all nodes are removed
        while soln.cost > self.max_cost {
            let Some((node_index, _)) = queue.pop() else {
                break;
            };

            let (prev_opt, next_opt) = path.neighbours(node_index);

            // compute before unlinking
            soln.cost -= self.removal_savings(&path, node_index);
            soln.score -= self.scores.get_score_by_index(node_index);

            path.remove(node_index);

            // only the two neighbours' savings change
            if let Some(prev) = prev_opt {
                queue.change_priority(&prev, priority(&path, prev));
            }
            if let Some(next) = next_opt {
                queue.change_priority(&next, priority(&path, next));
            }
        }

        soln.path = path.to_vec();
    }

    fn random_spanning_path<T: Rng>(&self, rng: &mut T) -> SolutionInfo {
        let node_count = self.scores.node_count();
        if node_count == 0 {
            return SolutionInfo {
                path: Vec::new(),
                score: 0.0,
                cost: 0.0,
            };
        }

        let mut unselected = Vec::from_iter(0..node_count);
        unselected.shuffle(rng);

        let mut path = Vec::with_capacity(self.scores.node_count());

        let mut total_cost = self.node_cost * node_count as f32;
        let mut total_score = 0.0;

        let mut prev_position = self.start_position;

        while let Some(node_index) = unselected.pop() {
            let node_position = self.scores.node_pos(node_index);
            total_cost += grid_dist(prev_position, node_position) * self.distance_multiplier;
            total_score += self.scores.get_score_by_index(node_index);

            path.push(node_index);
            prev_position = node_position;
        }

        SolutionInfo {
            path,
            score: total_score,
            cost: total_cost,
        }
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// designed to provide seeds for genetic solutions
    ///
    /// first generates a random, likely cost-exceeding path between nodes
    /// then, removes the least efficient entries until cost is no longer exceeded
    ///
    /// O(nlogn)
    fn greedy_random_subtractive_soln<T: Rng>(&self, rng: &mut T) -> SolutionInfo {
        let mut soln = self.random_spanning_path(rng);
        self.greedy_efficiency_path_shrink(&mut soln);
        soln
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
                .map(|i| MaxHeapEntry(self.scores.get_score_by_index(i), i)),
        );

        let mut remaining_cost = self.max_cost;

        let mut current_pos = self.start_position;

        // check the remaining cost *before* popping so no entry is discarded needlessly
        while remaining_cost > 0.0 {
            let Some(MaxHeapEntry(_, index)) = heap.pop() else {
                break;
            };

            let i_pos = self.scores.node_pos(index);
            let distance = grid_dist(current_pos, i_pos);
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

    /// modifies the given path either by shrinking to stay within max cost, or by expanding to
    /// use up remaining cost and increase score
    ///
    /// the path must not contain duplicate nodes, and its score and cost must be correct
    ///
    /// current method for shrinking is to iteratively remove the nodes that create the least
    /// score per unit of cost saved (see `greedy_efficiency_path_shrink`)
    ///
    /// current method for expansion is running greedy distance/prob algorithm
    fn greedy_path_adjust(&self, entry: &mut SolutionInfo) {
        if entry.cost > self.max_cost {
            self.greedy_efficiency_path_shrink(entry);
        }

        // only need to attempt expansion if a new node could feasibly be introduced.
        // distinct grid nodes are always at least distance 1 apart, so the cheapest possible
        // new node costs at least `node_cost + distance_multiplier`
        if entry.cost + self.node_cost + self.distance_multiplier < self.max_cost {
            self.greedy_score_distance_ratio_path_expand(entry)
        }
    }

    /// shrinks the given path to a valid cost by removing lowest-scoring nodes first,
    /// updating the entry's path, score and cost
    ///
    /// the path must not contain duplicate nodes, and its score and cost must be correct
    ///
    /// (currently unused; `greedy_efficiency_path_shrink` is preferred)
    #[allow(dead_code)]
    fn greedy_score_path_shrink(&self, entry: &mut SolutionInfo) {
        let mut path = LinkedPath::from_slice(&entry.path, self.scores.node_count());

        // construct a min heap of score
        let mut heap = BinaryHeap::from_iter(
            entry
                .path
                .iter()
                .map(|&i| MinHeapEntry(self.scores.get_score_by_index(i), i)),
        );

        // pop from score while invalid
        while entry.cost > self.max_cost {
            let Some(MinHeapEntry(_, node_index)) = heap.pop() else {
                break;
            };

            // compute before unlinking
            entry.cost -= self.removal_savings(&path, node_index);
            entry.score -= self.scores.get_score_by_index(node_index);

            path.remove(node_index);
        }

        entry.path = path.to_vec();
    }

    /// expands the given path based on score/distance greedy algorithm
    ///
    /// new nodes are appended to the end of the path, starting from the path's last node
    /// (or the start position if the path is empty)
    ///
    /// assumes the passed score and cost are correct and the path has no duplicate nodes
    ///
    /// updates the entry's path, score and cost
    fn greedy_score_distance_ratio_path_expand(&self, entry: &mut SolutionInfo) {
        let mut remaining_cost = self.max_cost - entry.cost;
        let mut current_pos: IVec2 = entry
            .path
            .last()
            .map_or_else(|| self.start_position, |x| self.scores.node_pos(*x));

        // construct bitset of unexplored nodes (all nodes not already on the path)
        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));
        for i in entry.path.iter() {
            unexplored.remove(*i);
        }

        while remaining_cost > 0.0 {
            if self.greedy_score_distance_ratio_soln_helper(
                &mut current_pos,
                &mut unexplored,
                &mut remaining_cost,
                &mut entry.score,
                &mut entry.path,
            ) {
                break;
            }
        }

        entry.cost = self.max_cost - remaining_cost;
    }

    /// recalculates the generation entry's score and cost
    fn update_score_and_cost(&self, entry: &mut SolutionInfo) {
        let (score, cost) = self.path_score_and_cost(&entry.path);
        entry.score = score;
        entry.cost = cost;
    }

    /// returns the path's score and cost in that order (score, cost)
    pub fn path_score_and_cost(&self, path: &[usize]) -> (f32, f32) {
        let mut score = 0.0;

        let node_cost = self.node_cost * path.len() as f32;
        let mut edge_cost = 0.0;

        let mut prev_pos = self.start_position;

        for curr in path.iter() {
            score += self.scores.get_score_by_index(*curr);

            let curr_pos = self.scores.node_pos(*curr);
            let dist = grid_dist(curr_pos, prev_pos);

            edge_cost += dist * self.distance_multiplier;

            prev_pos = curr_pos;
        }

        (score, node_cost + edge_cost)
    }

    /// Greedy algorithm that plans an exploration path between grid cells
    ///
    /// the next chosen cell is the one that maximizes (score/distance)
    ///
    /// returns (path, score)
    ///
    /// O(n^2)
    pub fn greedy_score_distance_ratio_soln(&self) -> SolutionInfo {
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
                break;
            }
        }

        SolutionInfo {
            path,
            score: total_score,
            cost: self.max_cost - remaining_cost,
        }
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
            let distance = grid_dist(*current_pos, i_pos);
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
                max_score = score;
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
    ///     - initial population: greedy random subtractive solutions (a random spanning path
    ///       shrunk by repeatedly removing the least efficient node until it fits the cost)
    ///     - end condition/output: a list of nodes representing visited nodes, after `passes`
    ///       generations
    ///     - the fitness function: score^2 / cost for positive-scoring individuals (rewards both
    ///       score and efficiency); non-positive scores rank below all positive ones
    ///     - crossover: heuristic crossover on disjoint pairs, inserting each parent's most
    ///       efficient substring into the other parent, followed by deduplication and greedy
    ///       shrinking/expansion so offspring are valid
    ///     - mutation: nudge random nodes to a nearby, unvisited grid cell; individuals pushed
    ///       over max cost by a mutation are greedily shrunk back to validity
    ///     - selection each generation:
    ///         - a few elites are preserved untouched
    ///         - top-performing individuals plus a random sample of the rest survive
    ///         - the population is refilled to its original size by breeding survivors
    ///     - output: the highest-scoring valid individual, after recomputing every individual's
    ///       score and cost to remove accumulated floating point drift
    pub fn genetic_soln(&self, generations: usize, population_size: usize) -> Vec<usize> {
        // trivial cases
        if self.max_cost <= 0.0 || self.scores.node_count() == 0 {
            return Vec::new();
        }

        let mut rng = rand::rng();

        let population_size = population_size.max(Self::MIN_POPULATION_SIZE);
        let elite_count = f32::ceil(population_size as f32 * Self::ELITE_FRACTION) as usize;
        let performance_count =
            f32::ceil(population_size as f32 * Self::PERFORMANCE_FRACTION) as usize;
        let diversity_count = f32::ceil(population_size as f32 * Self::DIVERSITY_FRACTION) as usize;

        // always leave at least one individual that does not pass, so the diversity sampling
        // below always has a non-empty range to draw from
        let pass_count = (performance_count + diversity_count).min(population_size - 1);
        let performance_count = performance_count.min(pass_count);
        let elite_count = elite_count.min(performance_count);

        // higher is better. non-positive scores are ranked by score (below every positive
        // individual) so that squaring cannot make negative scores look good
        let fitness = |soln: &SolutionInfo| {
            let v = if soln.score <= 0.0 {
                soln.score
            } else if soln.cost <= 0.0 {
                f32::INFINITY
            } else {
                soln.score * soln.score / soln.cost
            };
            if v.is_nan() { f32::NEG_INFINITY } else { v }
        };

        // get initial population
        let mut population = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            population.push(self.greedy_random_subtractive_soln(&mut rng));
        }

        for generation_num in 0..generations {
            if generation_num % Self::UNDRIFT_GENETIC_ITERATIONS == 0 {
                for soln in population.iter_mut() {
                    self.update_score_and_cost(soln);
                }
            }

            // cull population
            population.sort_unstable_by(|l, r| fitness(r).total_cmp(&fitness(l)));

            // copy elites to vector
            let elites: Vec<SolutionInfo> = population[0..elite_count].to_vec();

            // pick some random solutions for more genetic diversity
            for swap_index in performance_count..pass_count {
                let swap_with = rng.random_range(swap_index..population.len());
                population.swap(swap_index, swap_with);
            }

            // drop those that didnt pass
            population.truncate(pass_count);

            // crossover among the survivors
            self.genetic_crossover(&mut rng, &mut population);

            // refill the population back to its original size by breeding random survivors
            let target_size = population_size - elites.len();
            let parent_count = population.len();
            while population.len() < target_size {
                let p1 = rng.random_range(0..parent_count);
                let mut p2 = rng.random_range(0..parent_count);
                if parent_count > 1 && p2 == p1 {
                    p2 = (p1 + 1) % parent_count;
                }

                let [off1, off2] =
                    self.efficiency_heuristic_crossover(&mut rng, &population[p1], &population[p2]);
                population.push(off1);
                if population.len() < target_size {
                    population.push(off2);
                }
            }

            // mutate, then repair any individuals that mutation pushed over max cost
            for soln in population.iter_mut() {
                self.nudge_mutation(&mut rng, soln);
                if soln.cost > self.max_cost {
                    self.greedy_efficiency_path_shrink(soln);
                }
            }

            // re-introduce elites (never mutated)
            population.extend(elites);
        }

        // remove accumulated drift and make sure every individual is valid before choosing
        for soln in population.iter_mut() {
            self.update_score_and_cost(soln);
            while soln.cost > self.max_cost {
                self.greedy_efficiency_path_shrink(soln);
                self.update_score_and_cost(soln);
            }
        }

        // return the valid population member with greatest score
        population
            .into_iter()
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .map(|soln| soln.path)
            .unwrap_or_default()
    }

    /// performs genetic crossover on an entire generation
    ///
    /// the population is shuffled and paired into disjoint pairs, each of which crosses over
    /// with probability CROSSOVER_PROBABILITY (an odd individual out is left untouched)
    fn genetic_crossover<T: Rng>(&self, rng: &mut T, population: &mut Vec<SolutionInfo>) {
        population.shuffle(rng);
        for p1_idx in (0..population.len().saturating_sub(1)).step_by(2) {
            let p2_idx = p1_idx + 1;

            if rng.random_range(0.0..1.0_f32) >= Self::CROSSOVER_PROBABILITY {
                continue;
            }

            let [off1, off2] =
                self.efficiency_heuristic_crossover(rng, &population[p1_idx], &population[p2_idx]);

            population[p1_idx] = off1;
            population[p2_idx] = off2;
        }
    }

    /// detects the max-efficiency substring of nodes in a path
    /// returning the end-exclusive range
    ///
    /// efficiency of a substring is its total score divided by its total cost, where the cost
    /// is the node cost of each node plus the scaled edge cost between consecutive nodes *inside*
    /// the substring (the edge entering the substring is excluded, as it disappears once the
    /// substring is extracted and inserted elsewhere)
    ///
    /// maximizing a ratio is not directly solvable with Kadane's algorithm, so this uses
    /// Dinkelbach's method: given a candidate efficiency `lambda`, Kadane's algorithm finds the
    /// substring maximizing (score - lambda * cost); if that is positive, the substring beats
    /// `lambda`, so it becomes the new candidate. this converges to the exact optimum in a few
    /// iterations
    ///
    /// returns an empty range for an empty path
    ///
    /// O(n) per iteration
    ///
    /// used for heuristic crossover
    fn max_efficiency_substring(&self, path: &[usize]) -> Range<usize> {
        let n = path.len();
        if n == 0 {
            return 0..0;
        }

        // keeps the ratio finite if every cost is zero
        const MIN_COST: f32 = 1e-12;

        let scores: Vec<f32> = path
            .iter()
            .map(|&node| self.scores.get_score_by_index(node))
            .collect();

        // cost of extending a substring so that it includes node k (k >= 1)
        let mut extend_costs = vec![self.node_cost; n];
        for k in 1..n {
            let dist = grid_dist(
                self.scores.node_pos(path[k - 1]),
                self.scores.node_pos(path[k]),
            );
            extend_costs[k] = dist * self.distance_multiplier + self.node_cost;
        }

        let ratio = |range: &Range<usize>| -> f32 {
            let score: f32 = scores[range.clone()].iter().sum();
            let cost: f32 =
                self.node_cost + extend_costs[range.start + 1..range.end].iter().sum::<f32>();
            score / cost.max(MIN_COST)
        };

        // best substring maximizing sum(score - lambda * cost), and that sum
        let kadane = |lambda: f32| -> (Range<usize>, f32) {
            let mut best = 0..1;
            let mut best_value = f32::NEG_INFINITY;

            let mut cur_value = 0.0;
            let mut cur_start = 0;

            for k in 0..n {
                let start_value = scores[k] - lambda * self.node_cost;
                let extend_value = cur_value + scores[k] - lambda * extend_costs[k];

                if k == 0 || start_value >= extend_value {
                    cur_value = start_value;
                    cur_start = k;
                } else {
                    cur_value = extend_value;
                }

                if cur_value > best_value {
                    best_value = cur_value;
                    best = cur_start..k + 1;
                }
            }

            (best, best_value)
        };

        let mut best = 0..1;
        let mut lambda = ratio(&best);

        for _ in 0..Self::MAX_EFFICIENCY_ITERATIONS {
            let (candidate, value) = kadane(lambda);
            if value <= 1e-6 {
                break;
            }

            let candidate_ratio = ratio(&candidate);
            if candidate_ratio <= lambda {
                break;
            }

            best = candidate;
            lambda = candidate_ratio;
        }

        best
    }

    /// performs a heuristic double crossover using the max_efficiency_substring algorithm
    ///
    /// the first child inserts parent1's best genes on parent2's genome
    /// and the second child inserts parent2's best genes onto parent1's genome
    fn efficiency_heuristic_crossover<T: Rng>(
        &self,
        rng: &mut T,
        parent1: &SolutionInfo,
        parent2: &SolutionInfo,
    ) -> [SolutionInfo; 2] {
        let p1_best = self.max_efficiency_substring(&parent1.path);
        let p2_best = self.max_efficiency_substring(&parent2.path);

        [
            self.heuristic_crossover_insert_helper(
                rng,
                parent1,
                p1_best.clone(),
                parent2,
                p2_best.clone(),
            ),
            self.heuristic_crossover_insert_helper(rng, parent2, p2_best, parent1, p1_best),
        ]
    }

    /// performs a heuristic crossover using the max_efficiency_substring algorithm
    ///
    /// the child is a copy of dst_parent with src_parent's best genes (`src_range`) inserted at a
    /// random position that does not split dst_parent's own best genes (`dst_safe_range`),
    /// followed by deduplication and greedy adjustment so the child is valid
    fn heuristic_crossover_insert_helper<T: Rng>(
        &self,
        rng: &mut T,
        src_parent: &SolutionInfo,
        src_range: Range<usize>,
        dst_parent: &SolutionInfo,
        dst_safe_range: Range<usize>,
    ) -> SolutionInfo {
        // choose an insertion index in 0..=len that does not fall strictly inside the safe
        // range. valid indices are 0..=safe.start and safe.end..=len, which are
        // (safe.start + 1) + (len - safe.end + 1) choices in total
        let len = dst_parent.path.len();
        let before_count = dst_safe_range.start + 1;
        let after_count = len - dst_safe_range.end + 1;

        let choice = rng.random_range(0..before_count + after_count);
        let index = if choice < before_count {
            choice
        } else {
            dst_safe_range.end + (choice - before_count)
        };

        // insert src genes into that position
        let path: Vec<usize> = dst_parent.path[0..index]
            .iter()
            .copied()
            .chain(src_parent.path[src_range].iter().copied())
            .chain(dst_parent.path[index..len].iter().copied())
            .collect();

        let mut offspring = SolutionInfo {
            path,
            score: f32::NEG_INFINITY,
            cost: f32::INFINITY,
        };

        // ensure offspring is valid
        self.update_score_and_cost(&mut offspring);
        self.smart_dedup(&mut offspring);
        self.greedy_path_adjust(&mut offspring);

        offspring
    }

    /// replaces the node at `index` of the path with `new_node`, updating the solution's
    /// score and cost incrementally
    fn nudge_node(&self, soln: &mut SolutionInfo, index: usize, new_node: usize) {
        let old_node = soln.path[index];
        let old_pos = self.scores.node_pos(old_node);

        let new_pos = self.scores.node_pos(new_node);

        let prev_pos = if index == 0 {
            self.start_position
        } else {
            self.scores.node_pos(soln.path[index - 1])
        };

        // modify cost
        let distance_delta = if index + 1 == soln.path.len() {
            // last node: there is no outgoing edge
            grid_dist(prev_pos, new_pos) - grid_dist(prev_pos, old_pos)
        } else {
            let next_pos = self.scores.node_pos(soln.path[index + 1]);

            let init_dist = grid_dist(prev_pos, old_pos) + grid_dist(old_pos, next_pos);
            let final_dist = grid_dist(prev_pos, new_pos) + grid_dist(new_pos, next_pos);

            final_dist - init_dist
        };
        soln.cost += distance_delta * self.distance_multiplier;

        // modify score
        soln.score +=
            self.scores.get_score_by_index(new_node) - self.scores.get_score_by_index(old_node);

        // apply the change
        soln.path[index] = new_node;
    }

    /// mutates a path by moving a few random nodes to a nearby, unvisited grid cell
    ///
    /// mutations may cause the individual to exceed max cost; callers are responsible for
    /// repairing such individuals (see `genetic_soln`)
    fn nudge_mutation<T: Rng>(&self, rng: &mut T, entry: &mut SolutionInfo) {
        // record which grid nodes are already on the path
        let mut used = BitSet::<u64>::with_capacity_general(self.scores.node_count());
        for i in &entry.path {
            used.insert(*i);
        }

        // note that entry length will not change by nudging entries
        'gene_loop: for path_index in 0..entry.path.len() {
            if rng.random_range(0.0_f32..1.0_f32) >= Self::MUTATION_PROBABILITY {
                continue 'gene_loop;
            }

            let old_node = entry.path[path_index];
            let old_position = self.scores.node_pos(old_node);

            // attempt each possible nudge once, starting at a random one, until one is valid
            let start_delta = rng.random_range(0..Self::NUDGE_MUTATION_DELTAS.len());
            for offset in 0..Self::NUDGE_MUTATION_DELTAS.len() {
                let delta_index = (start_delta + offset) % Self::NUDGE_MUTATION_DELTAS.len();
                let new_position = old_position + Self::NUDGE_MUTATION_DELTAS[delta_index].coords;

                if !self.scores.contains_pos(new_position) {
                    continue;
                }

                let new_node = self.scores.node_index(new_position);
                if used.contains(new_node) {
                    continue;
                }

                self.nudge_node(entry, path_index, new_node);

                used.remove(old_node);
                used.insert(new_node);

                continue 'gene_loop;
            }
        }
    }

    /// 2-opt optimized greedy score-distance ratio solution
    pub fn optimized_greedy_score_distance_ratio_soln(
        &self,
        two_opt_compute_time: Duration,
    ) -> SolutionInfo {
        let mut soln = self.greedy_score_distance_ratio_soln();
        self.two_opt_soln(&mut soln, two_opt_compute_time);
        soln
    }

    /// applies 2-opt optimization to an existing solution
    /// performs up to the specified amount of passes as a proportion of path width
    fn two_opt_soln(&self, soln: &mut SolutionInfo, compute_time: Duration) {
        /// how often (in candidate moves) to check for exceeding compute time
        const TIME_CHECK_INTERVAL: u32 = 512;
        /// minimum improvement to accept a move (prevents cycling on float noise)
        const EPSILON: f32 = 1e-4;

        let n = soln.path.len();
        if n < 2 {
            return;
        }

        let end_time = Instant::now() + compute_time;
        let mut checks = 0u32;

        // position of tour index t, where t = 0 is the start and t >= 1 is path[t - 1]
        let pos = |path: &[usize], t: usize| -> IVec2 {
            if t == 0 {
                self.start_position
            } else {
                self.scores.node_pos(path[t - 1])
            }
        };

        let mut improved = true;
        while improved {
            improved = false;

            for i in 0..n {
                for j in i + 1..=n {
                    checks = checks.wrapping_add(1);
                    if checks % TIME_CHECK_INTERVAL == 0 && Instant::now() >= end_time {
                        return;
                    }

                    let a = pos(&soln.path, i);
                    let b = pos(&soln.path, i + 1);
                    let c = pos(&soln.path, j);

                    let mut delta = grid_dist(a, c) - grid_dist(a, b);
                    if j < n {
                        let d = pos(&soln.path, j + 1);
                        delta += grid_dist(b, d) - grid_dist(c, d);
                    }

                    if delta < -EPSILON {
                        soln.path[i..j].reverse();
                        soln.cost += delta * self.distance_multiplier;
                        improved = true;
                    }
                }
            }
        }
    }
}

// force evaluation of the compile-time sanity checks on the genetic algorithm's constants
const _: () = PathPlanningProblem::PASS_SIZE_ASSERTION;
const _: () = PathPlanningProblem::ELITE_FRACTION_ASSERTION;

/// a path stored as a doubly linked list inside a flat array indexed by grid node id,
/// allowing O(1) removal of a node
#[derive(Debug, Clone)]
struct LinkedPath {
    head: usize,
    tail: usize,
    /// (prev, next) links for every grid node id; (NONE, NONE) for nodes not on the path
    adj_map: Vec<(usize, usize)>,
    len: usize,
}

impl LinkedPath {
    const NONE: usize = usize::MAX;

    pub fn neighbours(&self, node_index: usize) -> (Option<usize>, Option<usize>) {
        let tuple = self.adj_map[node_index];
        (
            (tuple.0 != Self::NONE).then_some(tuple.0),
            (tuple.1 != Self::NONE).then_some(tuple.1),
        )
    }

    pub const fn len(&self) -> usize {
        self.len
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// the last node of the path, if any
    #[allow(dead_code)]
    pub fn tail(&self) -> Option<usize> {
        (self.tail != Self::NONE).then_some(self.tail)
    }

    pub fn iter(&self) -> impl Iterator<Item = usize> {
        let mut node = self.head;

        std::iter::from_fn(move || {
            if node != Self::NONE {
                let stored = node;
                node = self.adj_map[node].1;
                Some(stored)
            } else {
                None
            }
        })
    }

    /// removes the given node from the path
    ///
    /// the node must currently be on the path
    pub fn remove(&mut self, node_index: usize) {
        let neighbors = self.neighbours(node_index);
        self.adj_map[node_index] = (Self::NONE, Self::NONE);

        match neighbors {
            (Some(prev), Some(next)) => {
                self.adj_map[prev].1 = next;
                self.adj_map[next].0 = prev;
            }
            (Some(prev), None) => {
                self.adj_map[prev].1 = Self::NONE;
                self.tail = prev;
            }
            (None, Some(next)) => {
                self.adj_map[next].0 = Self::NONE;
                self.head = next;
            }
            (None, None) => {
                self.head = Self::NONE;
                self.tail = Self::NONE;
            }
        }

        self.len -= 1;
    }

    pub fn to_vec(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.len);

        for node in self.iter() {
            out.push(node)
        }

        out
    }

    /// builds a linked path from a slice of grid node ids
    ///
    /// `node_count` is the number of nodes in the grid, since links are indexed by grid node id
    /// (not by position in the slice). the slice must not contain duplicate nodes
    pub fn from_slice(slice: &[usize], node_count: usize) -> Self {
        let mut adj_map = vec![(Self::NONE, Self::NONE); node_count];

        if slice.is_empty() {
            return Self {
                head: Self::NONE,
                tail: Self::NONE,
                adj_map,
                len: 0,
            };
        }

        let head = slice[0];
        let mut prev = head;

        for &node in slice.iter().skip(1) {
            adj_map[prev].1 = node;
            adj_map[node].0 = prev;

            prev = node;
        }

        Self {
            head,
            tail: prev,
            adj_map,
            len: slice.len(),
        }
    }
}

/// a totally ordered f32 wrapper (via `f32::total_cmp`), so NaN can never cause a panic
#[derive(Debug, Clone, Copy)]
struct FPriority(pub f32);

impl PartialEq for FPriority {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for FPriority {}
impl PartialOrd for FPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

#[derive(Debug, Clone)]
pub struct SolutionInfo {
    pub path: Vec<usize>,
    pub score: f32,
    pub cost: f32,
}
