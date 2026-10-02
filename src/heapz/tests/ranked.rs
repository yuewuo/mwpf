mod common;

mod multi_pass_min {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap_after_pop(RankPairingHeap::multi_pass_min());
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap_after_pop(RankPairingHeap::multi_pass_min());
        }
    }

    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_min_heap(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_min_heap(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_smallest_to_largest_in_a_min_heap(RankPairingHeap::multi_pass_min());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_behind_a_lower_in_a_min_heap(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_before_a_higher_in_a_min_heap(RankPairingHeap::multi_pass_min());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_min_a_heap(RankPairingHeap::multi_pass_min());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::multi_pass_min());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::multi_pass_min());
        }
    }
}

mod multi_pass_min2 {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap_after_pop(RankPairingHeap::multi_pass_min2());
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap_after_pop(RankPairingHeap::multi_pass_min2());
        }
    }

    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_min_heap(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_min_heap(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_smallest_to_largest_in_a_min_heap(RankPairingHeap::multi_pass_min2());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_behind_a_lower_in_a_min_heap(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_before_a_higher_in_a_min_heap(RankPairingHeap::multi_pass_min2());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_min_a_heap(RankPairingHeap::multi_pass_min2());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::multi_pass_min2());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::multi_pass_min2());
        }
    }
}

mod single_pass_min {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap_after_pop(RankPairingHeap::single_pass_min());
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap_after_pop(RankPairingHeap::single_pass_min());
        }
    }

    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_min_heap(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_min_heap(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_smallest_to_largest_in_a_min_heap(RankPairingHeap::single_pass_min());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_behind_a_lower_in_a_min_heap(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_before_a_higher_in_a_min_heap(RankPairingHeap::single_pass_min());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_min_a_heap(RankPairingHeap::single_pass_min());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::single_pass_min());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::single_pass_min());
        }
    }
}

mod single_pass_min2 {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_min_heap_after_pop(
                RankPairingHeap::single_pass_min2(),
            );
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_min_heap_after_pop(
                RankPairingHeap::single_pass_min2(),
            );
        }
    }
    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_min_heap(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_min_heap(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_smallest_to_largest_in_a_min_heap(RankPairingHeap::single_pass_min2());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_behind_a_lower_in_a_min_heap(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_before_a_higher_in_a_min_heap(RankPairingHeap::single_pass_min2());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_min_a_heap(RankPairingHeap::single_pass_min2());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::single_pass_min2());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::single_pass_min2());
        }
    }
}

mod multi_pass_max {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap_after_pop(RankPairingHeap::multi_pass_max());
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap_after_pop(RankPairingHeap::multi_pass_max());
        }
    }

    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_max_heap(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_max_heap(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_largest_to_smallest_in_a_max_heap(RankPairingHeap::multi_pass_max());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_before_a_lower_in_a_max_heap(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_behind_a_higher_in_a_max_heap(RankPairingHeap::multi_pass_max());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_max_a_heap(RankPairingHeap::multi_pass_max());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::multi_pass_max());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::multi_pass_max());
        }
    }
}

mod multi_pass_max2 {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap_after_pop(RankPairingHeap::multi_pass_max2());
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap_after_pop(RankPairingHeap::multi_pass_max2());
        }
    }

    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_max_heap(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_max_heap(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_largest_to_smallest_in_a_max_heap(RankPairingHeap::multi_pass_max2());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_before_a_lower_in_a_max_heap(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_behind_a_higher_in_a_max_heap(RankPairingHeap::multi_pass_max2());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_max_a_heap(RankPairingHeap::multi_pass_max2());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::multi_pass_max2());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::multi_pass_max2());
        }
    }
}

mod single_pass_max {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap_after_pop(RankPairingHeap::single_pass_max());
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap_after_pop(RankPairingHeap::single_pass_max());
        }
    }

    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_max_heap(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_max_heap(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_largest_to_smallest_in_a_max_heap(RankPairingHeap::single_pass_max());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_before_a_lower_in_a_max_heap(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_behind_a_higher_in_a_max_heap(RankPairingHeap::single_pass_max());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_max_a_heap(RankPairingHeap::single_pass_max());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::single_pass_max());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::single_pass_max());
        }
    }
}

mod single_pass_max2 {
    mod delete {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn delete_an_element_by_key() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn delete_an_element_by_key_after_pop() {
            common::delete::will_delete_a_specific_element_by_key_from_max_heap_after_pop(
                RankPairingHeap::single_pass_max2(),
            );
        }
    }

    mod update {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn updates_an_element_by_key() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn updates_an_element_by_key_after_pop() {
            common::update::will_update_a_specific_element_by_key_in_a_max_heap_after_pop(
                RankPairingHeap::single_pass_max2(),
            );
        }
    }

    mod pop {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn removes_the_first_value_from_heap() {
            common::pop::removes_the_first_value_from_max_heap(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn returns_the_first_value_from_the_heap() {
            common::pop::returns_the_first_value_from_max_heap(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::pop::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn returns_all_elements_from_largest_to_smallest() {
            common::pop::returns_all_elements_from_largest_to_smallest_in_a_max_heap(RankPairingHeap::single_pass_max2());
        }
    }

    mod push {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn adds_a_value_to_the_heap() {
            common::push::adds_a_value_to_the_heap(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::push::adds_a_higher_item_to_the_heap_before_a_lower_in_a_max_heap(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn adds_a_lower_item_to_the_heap_before_a_higher() {
            common::push::adds_a_lower_item_to_the_heap_behind_a_higher_in_a_max_heap(RankPairingHeap::single_pass_max2());
        }
    }

    mod top {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_none_if_the_heap_is_empty() {
            common::top::returns_none_if_the_heap_is_empty(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn adds_a_higher_item_to_the_heap_behind_a_lower() {
            common::top::returns_the_first_value_in_max_a_heap(RankPairingHeap::single_pass_max2());
        }
    }

    mod size {
        use super::super::common;
        use mwpf::heapz::RankPairingHeap;

        #[test]
        fn returns_the_correct_size_of_a_heap_after_adding_elements() {
            common::size::returns_the_correct_size_of_a_heap_after_adding_elements(RankPairingHeap::single_pass_max2());
        }

        #[test]
        fn returns_the_first_value_in_a_heap() {
            common::size::returns_the_correct_size_of_a_heap_after_removing_an_element(RankPairingHeap::single_pass_max2());
        }
    }
}
