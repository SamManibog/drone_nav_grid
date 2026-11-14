use std::{cmp::Ordering, fmt::Debug, marker::PhantomData};

/// an entry of a max heap
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

pub trait Comparator {
    type Value;

    fn cmp(v1: &Self::Value, v2: &Self::Value) -> Ordering;
    fn eq(v1: &Self::Value, v2: &Self::Value) -> bool {
        Self::cmp(v1, v2).is_eq()
    }
}

/// an entry of a max heap that uses a different function to compare to entries
#[derive(Debug)]
pub struct MappedMaxHeapEntry<V, C: Comparator<Value = V>>{
    pub inner: V,
    pub phantom: PhantomData<C>,
}

impl<V, C: Comparator<Value = V>> MappedMaxHeapEntry<V, C> {
    pub fn new(inner: V) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<V: Clone, C: Comparator<Value = V>> Clone for MappedMaxHeapEntry<V, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom: PhantomData,
        }
    }
}
impl<V: Copy, C: Comparator<Value = V>> Copy for MappedMaxHeapEntry<V, C> {}

impl<V, C: Comparator<Value = V>> PartialEq for MappedMaxHeapEntry<V, C> {
    fn eq(&self, other: &Self) -> bool {
        C::eq(&self.inner, &other.inner)
    }
}
impl<V, C: Comparator<Value = V>> Eq for MappedMaxHeapEntry<V, C> {}
impl<V, C: Comparator<Value = V>> PartialOrd for MappedMaxHeapEntry<V, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}
impl<V, C: Comparator<Value = V>> Ord for MappedMaxHeapEntry<V, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        C::cmp(&self.inner, &other.inner)
    }
}

/// an entry of a min heap that uses a different function to compare to entries
#[derive(Debug)]
pub struct MappedMinHeapEntry<V, C: Comparator<Value = V>>{
    pub inner: V,
    pub phantom: PhantomData<C>,
}

impl<V, C: Comparator<Value = V>> MappedMinHeapEntry<V, C> {
    pub fn new(inner: V) -> Self {
        Self {
            inner,
            phantom: PhantomData,
        }
    }
}

impl<V: Clone, C: Comparator<Value = V>> Clone for MappedMinHeapEntry<V, C> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            phantom: PhantomData,
        }
    }
}
impl<V: Copy, C: Comparator<Value = V>> Copy for MappedMinHeapEntry<V, C> {}

impl<V, C: Comparator<Value = V>> PartialEq for MappedMinHeapEntry<V, C> {
    fn eq(&self, other: &Self) -> bool {
        C::eq(&self.inner, &other.inner)
    }
}
impl<V, C: Comparator<Value = V>> Eq for MappedMinHeapEntry<V, C> {}
impl<V, C: Comparator<Value = V>> PartialOrd for MappedMinHeapEntry<V, C> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(&other))
    }
}
impl<V, C: Comparator<Value = V>> Ord for MappedMinHeapEntry<V, C> {
    fn cmp(&self, other: &Self) -> Ordering {
        C::cmp(&self.inner, &other.inner).reverse()
    }
}
