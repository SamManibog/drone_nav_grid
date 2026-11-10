use std::cmp::Ordering;

/// an entry of a min heap
/// Ord is assumed to be satisfied for the set of used keys. however, only PartialOrd is enforced
/// Eq is assumed to be satisfied for the set of used keys. however, only PartialEq is enforced
#[derive(Debug, Clone, Copy)]
pub struct MaxHeapEntry<K: PartialOrd + PartialEq, V>(pub K, pub V);

impl<K: PartialOrd + PartialEq, V> PartialEq for MaxHeapEntry<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<K: PartialOrd + PartialEq, V> Eq for MaxHeapEntry<K, V> {}
impl<K: PartialOrd + PartialEq, V> PartialOrd for MaxHeapEntry<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}
impl<K: PartialOrd + PartialEq, V> Ord for MaxHeapEntry<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

/// an entry of a min heap
/// Ord is assumed to be satisfied for the set of used keys. however, only PartialOrd is enforced
/// Eq is assumed to be satisfied for the set of used keys. however, only PartialEq is enforced
#[derive(Debug, Clone, Copy)]
pub struct MinHeapEntry<K: PartialOrd + PartialEq, V>(pub K, pub V);

impl<K: PartialOrd + PartialEq, V> PartialEq for MinHeapEntry<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<K: PartialOrd + PartialEq, V> Eq for MinHeapEntry<K, V> {}

impl<K: PartialOrd + PartialEq, V> PartialOrd for MinHeapEntry<K, V> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}

impl<K: PartialOrd + PartialEq, V> Ord for MinHeapEntry<K, V> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap().reverse()
    }
}
