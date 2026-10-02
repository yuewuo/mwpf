use mwpf::heapz::{DecreaseKey, Heap};
use rand;
use rand::Rng;

#[derive(Hash, Copy, Clone, Eq, PartialEq, Debug)]
pub enum Element {
    Target,
    Node,
}

fn generate_numbers() -> Vec<i32> {
    let size = 1000;
    let mut rng = rand::thread_rng();
    (0..size).map(|_| rng.gen::<i32>()).collect()
}

pub mod pop {
    use super::{generate_numbers, Element, Heap};
    use std::cmp::{max, min};

    pub fn returns_the_first_value_from_min_heap<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut smallest = numbers[0];
        numbers.into_iter().for_each(|n| {
            smallest = min(smallest, n);
            let _ = &mut heap.push(n, n);
        });
        assert_eq!(heap.pop(), Some(smallest));
    }

    pub fn returns_the_first_value_from_max_heap<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut largest = numbers[0];
        numbers.into_iter().for_each(|n| {
            largest = max(largest, n);
            let _ = &mut heap.push(n, n);
        });
        assert_eq!(heap.pop(), Some(largest));
    }

    pub fn removes_the_first_value_from_min_heap<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        cloned.sort_by(|a, b| b.cmp(a));
        let _ = cloned.pop();
        let _ = heap.pop();
        assert_eq!(heap.top(), cloned.get(cloned.len() - 1));
    }

    pub fn removes_the_first_value_from_max_heap<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        let mut largest = numbers[0];
        let mut second_largest = largest;
        cloned.sort_by(|a, b| a.cmp(b));
        numbers.into_iter().for_each(|n| {
            second_largest = largest;
            largest = max(largest, n);
            let _ = &mut heap.push(n, n);
        });
        let _ = cloned.pop();
        let _ = heap.pop();
        assert_eq!(heap.top(), cloned.get(cloned.len() - 1));
    }

    pub fn returns_none_if_the_heap_is_empty<T: Heap<Element, i32>>(mut heap: T) {
        assert_eq!(heap.pop(), None);
    }

    pub fn returns_all_elements_from_smallest_to_largest_in_a_min_heap<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        cloned.sort_by(|a, b| b.cmp(a));
        while !cloned.is_empty() {
            assert_eq!(heap.pop(), cloned.pop());
        }
        assert_eq!(heap.pop(), None);
    }

    pub fn returns_all_elements_from_largest_to_smallest_in_a_max_heap<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        cloned.sort_by(|a, b| a.cmp(b));
        while !cloned.is_empty() {
            assert_eq!(heap.pop(), cloned.pop());
        }
        assert_eq!(heap.pop(), None);
    }
}

pub mod push {
    use super::{Element, Heap};

    pub fn adds_a_value_to_the_heap<T: Heap<Element, i32>>(mut heap: T) {
        let value = 1;
        let key = Element::Target;
        heap.push(key, value);
        assert_eq!(heap.top(), Some(&key));
    }

    pub fn adds_a_higher_item_to_the_heap_behind_a_lower_in_a_min_heap<T: Heap<Element, i32>>(mut heap: T) {
        let lower = 1;
        let higher = 2;
        heap.push(Element::Target, lower);
        heap.push(Element::Node, higher);
        assert_eq!(heap.top(), Some(&Element::Target));
    }

    pub fn adds_a_higher_item_to_the_heap_before_a_lower_in_a_max_heap<T: Heap<Element, i32>>(mut heap: T) {
        let lower = 1;
        let higher = 2;
        heap.push(Element::Node, lower);
        heap.push(Element::Target, higher);
        assert_eq!(heap.top(), Some(&Element::Target));
    }

    pub fn adds_a_lower_item_to_the_heap_before_a_higher_in_a_min_heap<T: Heap<Element, i32>>(mut heap: T) {
        let lower = 1;
        let higher = 2;
        heap.push(Element::Node, higher);
        heap.push(Element::Target, lower);
        assert_eq!(heap.top(), Some(&Element::Target));
    }

    pub fn adds_a_lower_item_to_the_heap_behind_a_higher_in_a_max_heap<T: Heap<Element, i32>>(mut heap: T) {
        let lower = 1;
        let higher = 2;
        heap.push(Element::Target, higher);
        heap.push(Element::Node, lower);
        assert_eq!(heap.top(), Some(&Element::Target));
    }
}

#[cfg(test)]
pub mod top {
    use super::{generate_numbers, Element, Heap};

    pub fn returns_the_first_value_in_min_a_heap<T: Heap<Element, i32>>(mut heap: T) {
        let mut numbers = generate_numbers();
        numbers.sort();
        numbers.reverse();
        let smallest = numbers.pop().unwrap();
        heap.push(Element::Target, smallest);
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(Element::Node, n);
        });
        assert_eq!(heap.top(), Some(&Element::Target));
    }

    pub fn returns_the_first_value_in_max_a_heap<T: Heap<Element, i32>>(mut heap: T) {
        let mut numbers = generate_numbers();
        numbers.sort();
        let largest = numbers.pop().unwrap();
        heap.push(Element::Target, largest);
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(Element::Node, n);
        });
        assert_eq!(heap.top(), Some(&Element::Target));
    }

    pub fn returns_none_if_the_heap_is_empty<T: Heap<Element, i32>>(heap: T) {
        assert_eq!(heap.top(), None);
    }
}

pub mod size {
    use super::{generate_numbers, Heap};

    pub fn returns_the_correct_size_of_a_heap_after_adding_elements<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let len = numbers.len();
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        assert_eq!(heap.size(), len);
    }

    pub fn returns_the_correct_size_of_a_heap_after_removing_an_element<T: Heap<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let len = numbers.len();
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        let _ = heap.pop();
        let _ = heap.pop();
        assert_eq!(heap.size(), len - 2);
    }
}

pub mod update {
    use super::{generate_numbers, DecreaseKey};
    use std::cmp::min;

    pub fn will_update_a_specific_element_by_key_in_a_min_heap<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let mut numbers = generate_numbers();
        let target = numbers.pop().unwrap();
        let mut cloned = numbers.clone();
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        cloned.sort_by(|a, b| b.cmp(a));
        let smallest = cloned[cloned.len() - 1];
        let next_smallest = smallest - 1;
        heap.update(&target, next_smallest);
        assert_eq!(heap.pop(), Some(target));
        while !cloned.is_empty() {
            assert_eq!(heap.pop(), cloned.pop());
        }
    }

    pub fn will_update_a_specific_element_by_key_in_a_min_heap_after_pop<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let mut numbers = generate_numbers();
        let mut cloned = numbers.clone();
        cloned.sort_by(|a, b| b.cmp(a));
        let target = cloned.remove(0);
        let index = numbers.iter().position(|n| n == &target).unwrap();
        numbers.remove(index);
        let mut smallest = target;
        numbers.into_iter().for_each(|n| {
            smallest = min(smallest, n);
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        let prev_smallest = smallest + 1;
        heap.update(&target, prev_smallest);
        assert_eq!(heap.pop(), cloned.pop());
        assert_eq!(heap.pop(), Some(target));
        while !cloned.is_empty() {
            assert_eq!(heap.pop(), cloned.pop());
        }
    }

    pub fn will_update_a_specific_element_by_key_in_a_max_heap<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let mut numbers = generate_numbers();
        let target = numbers.pop().unwrap();
        let mut cloned = numbers.clone();
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        cloned.sort_by(|a, b| a.cmp(b));
        let largest = cloned[cloned.len() - 1];
        let next_largest = largest + 1;
        heap.update(&target, next_largest);
        assert_eq!(heap.pop(), Some(target));
        while !cloned.is_empty() {
            assert_eq!(heap.pop(), cloned.pop());
        }
    }

    pub fn will_update_a_specific_element_by_key_in_a_max_heap_after_pop<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let mut numbers = generate_numbers();
        let mut cloned = numbers.clone();
        cloned.sort_by(|a, b| a.cmp(b));
        let target = cloned.remove(0);
        let index = numbers.iter().position(|n| n == &target).unwrap();
        numbers.remove(index);
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        let largest = cloned[cloned.len() - 1];
        let prev_largest = largest - 1;
        heap.update(&target, prev_largest);
        assert_eq!(heap.pop(), cloned.pop());
        assert_eq!(heap.pop(), Some(target));
        while !heap.is_empty() {
            assert_eq!(heap.pop(), cloned.pop());
        }
    }
}

pub mod delete {
    use super::{generate_numbers, DecreaseKey};

    pub fn will_delete_a_specific_element_by_key_from_min_heap<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        cloned.sort_by(|a, b| b.cmp(a));
        let target = cloned[0] + 100;
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        heap.delete(&target);
        while !cloned.is_empty() && !heap.is_empty() {
            assert_eq!(heap.pop(), cloned.pop())
        }
    }

    pub fn will_delete_a_specific_element_by_key_from_min_heap_after_pop<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        cloned.sort_by(|a, b| b.cmp(a));
        let target = cloned[0] + 100;
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        assert_eq!(heap.pop(), cloned.pop());
        heap.delete(&target);
        while !cloned.is_empty() && !heap.is_empty() {
            assert_eq!(heap.pop(), cloned.pop())
        }
    }

    pub fn will_delete_a_specific_element_by_key_from_max_heap<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        cloned.sort_by(|a, b| a.cmp(b));
        let target = cloned[0] - 100;
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        heap.delete(&target);
        while !cloned.is_empty() && !heap.is_empty() {
            assert_eq!(heap.pop(), cloned.pop())
        }
    }

    pub fn will_delete_a_specific_element_by_key_from_max_heap_after_pop<T: DecreaseKey<i32, i32>>(mut heap: T) {
        let numbers = generate_numbers();
        let mut cloned = numbers.clone();
        cloned.sort_by(|a, b| a.cmp(b));
        let target = cloned[0] - 100;
        numbers.into_iter().for_each(|n| {
            let _ = &mut heap.push(n, n);
        });
        heap.push(target, target);
        assert_eq!(heap.pop(), cloned.pop());
        heap.delete(&target);
        while !cloned.is_empty() && !heap.is_empty() {
            assert_eq!(heap.pop(), cloned.pop())
        }
    }
}
