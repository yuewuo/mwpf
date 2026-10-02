mod common;

mod pop {
    use super::common;
    use mwpf::heapz::PairingHeap;

    #[test]
    fn returns_the_first_value_from_the_min_heap() {
        common::pop::returns_the_first_value_from_min_heap(PairingHeap::min());
    }

    #[test]
    fn returns_the_first_value_from_the_max_heap() {
        common::pop::returns_the_first_value_from_max_heap(PairingHeap::max());
    }

    #[test]
    fn removes_the_first_value_from_min_heap() {
        common::pop::removes_the_first_value_from_min_heap(PairingHeap::min());
    }

    #[test]
    fn removes_the_first_value_from_max_heap() {
        common::pop::removes_the_first_value_from_max_heap(PairingHeap::max());
    }

    #[test]
    fn returns_none_if_the_min_heap_is_empty() {
        common::pop::returns_none_if_the_heap_is_empty(PairingHeap::min());
    }

    #[test]
    fn returns_none_if_the_max_heap_is_empty() {
        common::pop::returns_none_if_the_heap_is_empty(PairingHeap::max());
    }

    #[test]
    fn returns_all_elements_from_smallest_to_largest_in_a_min_heap() {
        common::pop::returns_all_elements_from_smallest_to_largest_in_a_min_heap(PairingHeap::min());
    }

    #[test]
    fn returns_all_elements_from_largest_to_smallest_in_a_max_heap() {
        common::pop::returns_all_elements_from_largest_to_smallest_in_a_max_heap(PairingHeap::max());
    }
}

mod push {
    use super::common;
    use mwpf::heapz::PairingHeap;

    #[test]
    fn adds_a_value_to_the_heap() {
        common::push::adds_a_value_to_the_heap(PairingHeap::min());
    }

    #[test]
    fn adds_a_higher_item_to_the_heap_behind_a_lower_in_a_min_heap() {
        common::push::adds_a_higher_item_to_the_heap_behind_a_lower_in_a_min_heap(PairingHeap::min());
    }

    #[test]
    fn adds_a_higher_item_to_the_heap_before_a_lower_in_a_max_heap() {
        common::push::adds_a_higher_item_to_the_heap_before_a_lower_in_a_max_heap(PairingHeap::max());
    }

    #[test]
    fn adds_a_lower_item_to_the_heap_before_a_higher_in_a_min_heap() {
        common::push::adds_a_lower_item_to_the_heap_before_a_higher_in_a_min_heap(PairingHeap::min());
    }

    #[test]
    fn adds_a_lower_item_to_the_heap_behind_a_higher_in_a_max_heap() {
        common::push::adds_a_lower_item_to_the_heap_behind_a_higher_in_a_max_heap(PairingHeap::max());
    }
}

mod top {
    use super::common;
    use mwpf::heapz::PairingHeap;

    #[test]
    fn returns_the_first_value_in_a_max_heap() {
        common::top::returns_the_first_value_in_max_a_heap(PairingHeap::max());
    }

    #[test]
    fn returns_the_first_value_in_a_min_heap() {
        common::top::returns_the_first_value_in_min_a_heap(PairingHeap::min());
    }

    #[test]
    fn returns_none_if_the_heap_is_empty() {
        common::top::returns_none_if_the_heap_is_empty(PairingHeap::max());
    }
}

mod size {
    use super::common;
    use mwpf::heapz::PairingHeap;

    #[test]
    fn returns_the_correct_size_of_a_heap_after_adding_elements() {
        common::size::returns_the_correct_size_of_a_heap_after_adding_elements(PairingHeap::max());
    }

    #[test]
    fn returns_the_correct_size_of_a_heap_after_removing_an_element() {
        common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(PairingHeap::min());
    }
}
