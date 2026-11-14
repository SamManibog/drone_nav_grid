use std::{cmp::Ordering, collections::BinaryHeap, ops::Range};

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
    /// the number of genetic iterations to run before recalculating scores and costs
    const UNDRIFT_GENETIC_ITERATIONS: usize = 10;

    /// the initial population size as a proportion of maximum possible gene count
    const POPULATION_SIZE_MULTIPLIER: f32 = 8.0;

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

    const PASS_FRACTION: f32 =
        Self::ELITE_FRACTION + Self::PERFORMANCE_FRACTION + Self::DIVERSITY_FRACTION;

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

    /// smart deduplication step for a solution with duplicate nodes
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

        let comparative_path_idx_savings = |idx: usize| -> f32 {
            let pos = self.scores.node_pos(soln.path[idx]);

            let prev_pos = if idx == 0 {
                self.start_position
            } else {
                self.scores.node_pos(soln.path[idx - 1])
            };

            if idx == soln.path.len() - 1 {
                grid_dist(prev_pos, pos)
            } else {
                let next_pos = self.scores.node_pos(soln.path[idx + 1]);
                grid_dist(prev_pos, pos) + grid_dist(pos, next_pos) - grid_dist(prev_pos, next_pos)
            }
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
    /// O(n^2)
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

        // smart dedup on specific node
        for dupe_node in dupes {
            self.smart_dedup_node_helper(soln, dupe_node);
        }
    }

    /// shrinks a solution until it is valid, prioritizing the removal of inefficient edges
    fn greedy_efficiency_path_shrink(&self, soln: &mut SolutionInfo) {
        let mut path = LinkedPath::from_slice(&soln.path);

        // s(i) = score of node i
        // c(i, j) = cost of edge between i and j
        // r = root/start position
        // C = node cost
        //
        // Cost saved by removing node i given the current links. Handles head
        // (prev == NONE), tail (next == NONE), and a lone node (both NONE).
        //
        // let path p = x, ..., y, ..., z
        //   head:  c(r, x) + c(x, x + 1) - c(r, x + 1) + C
        //   inner: c(y - 1, y) + c(y, y + 1) - c(y - 1, y + 1) + C
        //   tail:  c(z - 1, z) + C
        //
        // (distances are scaled by the distance multiplier)
        let savings = |adj: &LinkedPath, i: usize| -> f32 {
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
        };

        // efficiency = score created per unit of cost that removal would save.
        // The queue is a max-heap, so negate: the least efficient node pops first.
        // Removals that save nothing are treated as infinitely efficient (never
        // chosen early).
        let priority = |adj: &LinkedPath, i: usize| -> FPriority {
            let saved = savings(adj, i);
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
            soln.cost -= savings(&path, node_index);
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

        soln.path = path.iter().collect();
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
        let mut soln = Self::random_spanning_path(&self, rng);
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
                .into_iter()
                .map(|i| MaxHeapEntry(self.scores.get_score_by_index(i), i)),
        );

        let mut remaining_cost = self.max_cost;

        let mut current_pos = self.start_position;

        while let Some(MaxHeapEntry(_, index)) = heap.pop()
            && remaining_cost > 0.0
        {
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

    /// modifies the given path either by expanding to maximize score, or shrinking to stay within
    /// max score
    ///
    /// DOES NOT PERFORM DUPLICATE DETECTION/REMOVAL
    ///
    /// current method for shrinking is to iteratively select and remove the lowest-scoring included
    /// nodes
    ///
    /// current method for expansion is running greedy distance/prob algorithm
    fn greedy_path_adjust(&self, entry: &mut SolutionInfo) {
        if entry.cost > self.max_cost {
            Self::greedy_efficiency_path_shrink(&self, entry);
        }

        // only need to attempt expansion if a new node could feasibly be introduced
        if entry.cost + self.node_cost + self.distance_multiplier < self.max_cost {
            Self::greedy_score_distance_ratio_path_expand(&self, entry)
        }
    }

    /// shrinks the given path to a valid cost by removing lowest-scoring nodes first
    ///
    /// returns the new score
    fn greedy_score_path_shrink(&self, entry: &mut SolutionInfo) {
        // construct a min heap of score
        let mut heap = BinaryHeap::from_iter(
            entry
                .path
                .iter()
                .map(|i| MinHeapEntry(self.scores.get_score_by_index(*i), *i)),
        );

        // pop from score while invalid
        while let Some(node) = heap.pop()
            && entry.cost > self.max_cost
        {
            let i = node.1;

            let node_pos = self.scores.node_pos(i);

            // adjust path cost
            if i == 0 && i == entry.path.len() {
                // resulting path is empty. no cost, no score
                entry.path.pop();
                entry.cost = 0.0;
                entry.score = 0.0;
                return;
            } else if i == entry.path.len() - 1 {
                entry.cost -= grid_dist(self.scores.node_pos(i - 1), node_pos)
            } else {
                let prev_pos = if i == 0 {
                    self.start_position
                } else {
                    self.scores.node_pos(i - 1)
                };

                let next_pos = self.scores.node_pos(i + 1);

                let original_dist = grid_dist(prev_pos, node_pos) + grid_dist(node_pos, next_pos);
                let new_dist = grid_dist(prev_pos, next_pos);

                entry.cost -= (new_dist - original_dist) * self.distance_multiplier;
            };

            // adjust score
            entry.score -= self.scores.get_score_by_index(i);

            // remove from path
            entry.path.remove(i);
        }
    }

    /// expands the given path based on score/distance greedy algorithm
    /// assumes the path is non-empty
    ///
    /// assumes the passed total score and cost are correct
    ///
    /// returns the new score
    fn greedy_score_distance_ratio_path_expand(&self, entry: &mut SolutionInfo) {
        let mut remaining_cost = self.max_cost - entry.cost;
        let mut current_pos: IVec2 = entry
            .path
            .first()
            .map_or_else(|| self.start_position, |x| self.scores.node_pos(*x));

        // construct bitset
        let mut unexplored: BitSet<u64> =
            BitSet::from_bit_vec_general(BitVec::from_elem_general(self.scores.node_count(), true));
        for i in entry.path.iter() {
            unexplored.insert(*i);
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
    pub fn genetic_soln(&self, passes: usize) -> Vec<usize> {
        // trivial case
        if self.max_cost <= 0.0 {
            return Vec::new();
        }

        let mut rng = rand::rng();

        let max_gene_count = (self.scores.node_count() as f32)
            .max(self.max_cost / (self.distance_multiplier + self.node_cost));

        let population_size = (max_gene_count as f32 * Self::POPULATION_SIZE_MULTIPLIER) as usize;
        let elite_count = f32::ceil(population_size as f32 * Self::ELITE_FRACTION) as usize;
        let performance_count =
            f32::ceil(population_size as f32 * Self::PERFORMANCE_FRACTION) as usize;
        let diversity_count = f32::ceil(population_size as f32 * Self::DIVERSITY_FRACTION) as usize;
        let pass_count = performance_count + diversity_count;

        let fitness = |soln: &SolutionInfo| soln.score * soln.score / soln.cost;

        // get initial population
        let mut population = Vec::with_capacity(population_size);
        for _ in 0..population_size {
            population.push(self.greedy_random_subtractive_soln(&mut rng));
        }

        for generation_num in 0..passes {
            if generation_num % Self::UNDRIFT_GENETIC_ITERATIONS == 0 {
                for soln in population.iter_mut() {
                    self.update_score_and_cost(soln);
                }
            }

            let mut elites: Vec<SolutionInfo>;
            // cull population
            {
                population
                    .sort_unstable_by(|l, r| (fitness)(r).partial_cmp(&(fitness)(l)).unwrap());

                // copy elites to vector
                elites = population[0..elite_count].iter().cloned().collect();

                // pick some random solutions for more genetic diversity
                for swap_index in performance_count..pass_count {
                    let swap_with = rng.random_range(swap_index..population.len());
                    population.swap(swap_index, swap_with);
                }

                // drain those that didnt pass
                population.drain(pass_count..population.len());
            }

            // perform crossover and mutation
            self.genetic_crossover(&mut rng, &mut population);
            for soln in population.iter_mut() {
                self.nudge_mutation(&mut rng, soln);
            }

            // re-introduce elites
            population.append(&mut elites);
        }

        // return population member with greatest score
        population
            .into_iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
            .unwrap()
            .path
    }

    /// performs genetic crossover on an entire generation
    fn genetic_crossover<T: Rng>(&self, rng: &mut T, population: &mut Vec<SolutionInfo>) {
        population.shuffle(rng);
        for (p1_idx, p2_idx) in (1..population.len()).into_iter().map(|i| (i - 1, i)) {
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
    /// used for heuristic crossover
    ///
    /// O(n)
    fn max_efficiency_substring(&self, path: &Vec<usize>) -> Range<usize> {
        let mut local_max_efficiency;
        let mut local_max_cost = f32::INFINITY;
        let mut local_max_score = f32::NEG_INFINITY;
        let mut local_max_start = 0;
        let mut local_max_len = 0;

        let mut max_efficiency = 0.0;
        let mut max_start = 0;
        let mut max_len = 0;

        let mut prev_pos = self.start_position;
        for (idx, node) in path.iter().copied().enumerate() {
            let pos = self.scores.node_pos(node);

            let as_new_cost = grid_dist(prev_pos, pos) * self.distance_multiplier + self.node_cost;
            let as_new_score = self.scores.get_score_by_index(node);
            let as_new_efficiency = as_new_score / as_new_cost;

            let with_new_cost = local_max_cost + as_new_cost;
            let with_new_score = local_max_score + as_new_score;
            let with_new_efficiency = with_new_score / with_new_cost;

            if as_new_efficiency > with_new_efficiency {
                local_max_efficiency = as_new_efficiency;
                local_max_score = as_new_score;
                local_max_cost = as_new_cost;
                local_max_start = idx;
                local_max_len = 1;
            } else {
                local_max_efficiency = with_new_efficiency;
                local_max_score = with_new_score;
                local_max_cost = with_new_cost;
                local_max_len += 1;
            }

            if local_max_efficiency > max_efficiency {
                max_efficiency = local_max_efficiency;
                max_start = local_max_start;
                max_len = local_max_len;
            }

            prev_pos = pos;
        }

        max_start..max_start + max_len
    }

    /// performs a heuristic double crossover using the max_efficiency_substring algorithm
    ///
    /// the child inserts parent1's best genes on parent2's genome
    /// and parent2's best genes onto parent1's genome
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
    /// the child inserts parent1's best genes on parent2's genes
    fn heuristic_crossover_insert_helper<T: Rng>(
        &self,
        rng: &mut T,
        src_parent: &SolutionInfo,
        src_range: Range<usize>,
        dst_parent: &SolutionInfo,
        dst_safe_range: Range<usize>,
    ) -> SolutionInfo {
        // select random position for p1 genes that doesnt overlap good range
        let index = {
            let raw = rng.random_range(0..dst_parent.path.len());
            if raw > dst_safe_range.start {
                raw + dst_safe_range.end
            } else {
                raw
            }
        };

        // insert p1 genes into that position
        let path: Vec<usize> = dst_parent.path[0..index]
            .iter()
            .copied()
            .chain(src_parent.path[src_range].iter().copied())
            .chain(
                dst_parent.path[index..dst_parent.path.len()]
                    .iter()
                    .copied(),
            )
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
        let distance_delta = if index == soln.path.len() {
            grid_dist(prev_pos, new_pos) - grid_dist(prev_pos, old_pos)
        } else {
            let next_pos = self.scores.node_pos(soln.path[index + 1]);

            let init_dist = grid_dist(prev_pos, old_pos) + grid_dist(old_pos, next_pos);
            let final_dist = grid_dist(prev_pos, new_pos) + grid_dist(new_pos, next_pos);

            final_dist - init_dist
        };
        soln.cost += distance_delta * self.distance_multiplier;

        // modify score
        soln.score += self.scores.get_score_by_index(new_node) - self.scores.get_score_by_index(old_node);
    }

    /// mutates a path by moving a few random nodes by a random amount
    /// mutations may cause the individual to exceed max cost
    fn nudge_mutation<T: Rng>(&self, rng: &mut T, entry: &mut SolutionInfo) {
        // add contained nodes to range
        let mut used = BitSet::<u64>::with_capacity_general(entry.path.len());
        for i in &entry.path {
            used.insert(*i);
        }

        // note that entry length will not change by nudging entries
        'gene_loop: for node_index in 0..entry.path.len() {
            let random_var: f32 = rng.random_range(0.0_f32..=1.0_f32);
            if random_var > Self::MUTATION_PROBABILITY {
                continue 'gene_loop;
            }

            let node_position = self.scores.node_pos(node_index);

            let mut nudge_index = rng.random_range(0..Self::NUDGE_MUTATION_DELTAS.len());
            let stop_index = nudge_index;

            // attempt to nudge for all possible nudges
            'nudge_attempt_loop: {
                let new_position = node_position + Self::NUDGE_MUTATION_DELTAS[nudge_index].coords;
                if self.scores.contains_pos(node_position) {
                    let new_node_index = self.scores.node_index(new_position);
                    if !used.contains(new_node_index) {
                        self.nudge_node(entry, node_index, new_node_index);

                        used.remove(node_index);
                        used.insert(new_node_index);

                        break 'nudge_attempt_loop;
                    }
                }

                nudge_index += 1;
                nudge_index %= Self::NUDGE_MUTATION_DELTAS.len();

                if nudge_index == stop_index {
                    break 'nudge_attempt_loop;
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
struct LinkedPath {
    head: usize,
    tail: usize,
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
    pub fn remove(&mut self, node_index: usize) {
        let neighbors = self.neighbours(node_index);
        self.adj_map[node_index] = (Self::NONE, Self::NONE);

        match neighbors {
            (Some(prev), Some(next)) => {
                self.adj_map[prev].1 = next;
                self.adj_map[next].0 = next;
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

    pub fn from_slice(slice: &[usize]) -> Self {
        if slice.len() == 0 {
            return Self {
                head: Self::NONE,
                tail: Self::NONE,
                adj_map: Vec::new(),
                len: 0,
            };
        }

        let head = slice[0];
        let mut tail = head;
        let mut prev = head;

        let mut adj_map = vec![(Self::NONE, Self::NONE); slice.len()];

        for node in slice.iter().skip(1).cloned() {
            adj_map[node].0 = prev;

            prev = node;
            tail = node;
        }

        Self {
            head,
            tail,
            adj_map,
            len: slice.len(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct FPriority(pub f32);

impl PartialEq for FPriority {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
    fn ne(&self, other: &Self) -> bool {
        self.0 != other.0
    }
}
impl Eq for FPriority {}
impl PartialOrd for FPriority {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}
impl Ord for FPriority {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[derive(Debug, Clone)]
struct SolutionInfo {
    path: Vec<usize>,
    score: f32,
    cost: f32,
}

/// colleciton of fitness functions
#[allow(unused)]
mod fitness {
    pub fn score_efficiency(score: f32, cost: f32) -> f32 {
        score * score / cost
    }
}
