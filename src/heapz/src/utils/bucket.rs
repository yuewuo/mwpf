use crate::heapz::utils::math::log;

pub struct Bucket<V: Clone> {
    store: Vec<Vec<V>>,
}

impl<V: Clone> Bucket<V> {
    pub fn new(size: usize) -> Self {
        let fill_size =
            (((if size > 0 { log(size) } else { 0 }) + 1) as f32 * 1.4).floor() as usize;
        Bucket {
            store: vec![vec![]; fill_size],
        }
    }

    pub fn insert(&mut self, key: usize, value: V) {
        self.store[key].push(value);
    }

    pub fn contains_key(&self, key: usize) -> bool {
        !self.store[key].is_empty()
    }

    pub fn remove(&mut self, key: usize) -> Option<V> {
        self.store[key].pop()
    }

    pub fn drain(self) -> impl Iterator<Item = V> {
        self.store
            .into_iter()
            .filter(|bucket| !bucket.is_empty())
            .map(|mut bucket| bucket.pop().unwrap())
    }
}
