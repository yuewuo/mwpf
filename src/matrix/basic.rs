use super::interface::*;
use super::row::*;
use super::visualize::*;
use crate::util::*;
use derivative::Derivative;
use crate::dual_module_pq::{EdgeWeak, VertexWeak};

#[derive(Clone, Derivative, PartialEq, Eq)]
#[derivative(Default(new = "true"))]
pub struct BasicMatrix {
    /// the vertices already maintained by this parity check
    pub vertices: FastIterSet<VertexWeak>,
    /// the edges maintained by this parity check, mapping to the local indices
    pub edges: FastIterMap<EdgeWeak, VarIndex>,
    /// variable index map to edge index
    pub variables: Vec<EdgeWeak>,
    pub constraints: Vec<ParityRow>,
}

impl MatrixBasic for BasicMatrix {
    fn add_variable(&mut self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        if self.edges.contains_key(&edge_weak.clone()) {
            // variable already exists
            return None;
        }
        let var_index = self.variables.len();
        self.edges.insert(edge_weak.clone(), var_index);
        self.variables.push(edge_weak.clone());
        ParityRow::add_one_variable(&mut self.constraints, self.variables.len());
        Some(var_index)
    }

    fn add_constraint(
        &mut self,
        vertex_weak: VertexWeak,
        incident_edges: &[EdgeWeak],
        parity: bool,
    ) -> Option<Vec<VarIndex>> {
        if self.vertices.contains(&vertex_weak) {
            // no need to add repeat constraint
            return None;
        }
        let mut var_indices = None;
        self.vertices.insert(vertex_weak.clone());
        for edge_weak in incident_edges.iter() {
            if let Some(var_index) = self.add_variable(edge_weak.clone()) {
                // this is a newly added edge
                var_indices.get_or_insert_with(Vec::new).push(var_index);
            }
        }
        let mut row = ParityRow::new_length(self.variables.len());
        for edge_weak in incident_edges.iter() {
            let var_index = self.edges[&edge_weak.clone()];
            row.set_left(var_index, true);
        }
        row.set_right(parity);
        self.constraints.push(row);
        var_indices
    }

    /// row operations
    fn xor_row(&mut self, target: RowIndex, source: RowIndex) {
        ParityRow::xor_two_rows(&mut self.constraints, target, source)
    }

    fn swap_row(&mut self, a: RowIndex, b: RowIndex) {
        self.constraints.swap(a, b);
    }

    fn get_lhs(&self, row: RowIndex, var_index: VarIndex) -> bool {
        self.constraints[row].get_left(var_index)
    }

    fn get_rhs(&self, row: RowIndex) -> bool {
        self.constraints[row].get_right()
    }

    fn var_to_edge_index(&self, var_index: VarIndex) -> EdgeWeak {
        self.variables[var_index].clone()
    }

    fn edge_to_var_index(&self, edge_weak: EdgeWeak) -> Option<VarIndex> {
        self.edges.get(&edge_weak.clone()).cloned()
    }

    fn get_vertices(&self) -> FastIterSet<VertexWeak> {
        self.vertices.clone()
    }

    fn get_edges(&self) -> FastIterSet<EdgeWeak> {
        self.edges.keys().cloned().collect()
    }
}

impl MatrixView for BasicMatrix {
    fn columns(&mut self) -> usize {
        self.variables.len()
    }

    fn column_to_var_index(&self, column: ColumnIndex) -> VarIndex {
        column
    }

    fn rows(&mut self) -> usize {
        self.constraints.len()
    }
}

impl VizTrait for BasicMatrix {
    fn viz_table(&mut self) -> VizTable {
        VizTable::from(self)
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dual_module_pq::{Edge, EdgePtr, Vertex, VertexPtr};
    use crate::num_traits::Zero;
    use std::collections::HashSet;

    /// Helper to create mock pointers for testing
    pub fn initialize_vertex_edges_for_matrix_testing(
        vertex_indices: Vec<VertexIndex>,
        edge_indices: Vec<EdgeIndex>,
    ) -> (Vec<VertexPtr>, Vec<EdgePtr>) {
        let edges: Vec<EdgePtr> = edge_indices
            .into_iter()
            .map(|edge_index| {
                EdgePtr::new_value(
                    Edge {
                        edge_index,
                        weight: Rational::zero(),
                        dual_nodes: vec![],
                        vertices: vec![],
                        last_updated_time: Rational::zero(),
                        growth_at_last_updated_time: Rational::zero(),
                        grow_rate: Rational::zero(),
                        // unit_index: Some(0),
                        // connected_to_boundary_vertex: false,
                        #[cfg(feature = "incr_lp")]
                        cluster_weights: hashbrown::HashMap::new(),
                    },
                    (0, edge_index),
                )
            })
            .collect();

        let vertices: Vec<VertexPtr> = vertex_indices
            .into_iter()
            .map(|vertex_index| {
                VertexPtr::new_value(
                    Vertex {
                        vertex_index,
                        is_defect: false,
                        edges: vec![],
                        // mirrored_vertices: vec![],
                    },
                    (0, vertex_index),
                )
            })
            .collect();

        (vertices, edges)
    }

    pub fn edge_vec_from_indices(edge_sequences: &[usize], edges: &Vec<EdgePtr>) -> Vec<EdgeWeak> {
        edge_sequences
            .iter()
            .map(|&i| edges[i].downgrade())
            .collect()
    }

    #[test]
    fn basic_matrix_1() {
        let mut matrix = BasicMatrix::new();
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(
            vec![0, 1, 2], 
            vec![1, 4, 12, 345]
        );

        // Add variables
        for edge in edges.iter() {
            matrix.add_variable(edge.downgrade());
        }

        // Add constraints using sequence indices from the 'edges' vector
        matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&[0, 1, 2], &edges), true);
        matrix.add_constraint(vertices[1].downgrade(), &edge_vec_from_indices(&[1, 3], &edges), false);
        matrix.add_constraint(vertices[2].downgrade(), &edge_vec_from_indices(&[0, 3], &edges), true);

        matrix.printstd();
        
        let vertex_indices: HashSet<_> = matrix.get_vertices().iter()
            .map(|v| v.upgrade_force().read_recursive().vertex_index).collect();
        assert_eq!(vertex_indices, [0, 1, 2].into_iter().collect());
    }

    #[test]
    fn basic_matrix_should_not_add_repeated_constraint() {
        let mut matrix = BasicMatrix::new();
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(
            vec![0, 1], 
            vec![1, 4, 8]
        );

        // First add: Success (returns indices of newly created variables 0, 1, 2)
        assert_eq!(
            matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&[0, 1, 2], &edges), false),
            Some(vec![0, 1, 2])
        );

        // Second add (new vertex): Success (returns None because no NEW variables were created)
        assert_eq!(
            matrix.add_constraint(vertices[1].downgrade(), &edge_vec_from_indices(&[1, 2], &edges), true),
            None
        );

        // Third add (repeat vertex): Success is None because vertex already exists
        assert_eq!(
            matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&[1], &edges), true),
            None
        );
    }

    #[test]
    fn basic_matrix_row_operations() {
        let mut matrix = BasicMatrix::new();
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(
            vec![0, 1, 2], 
            vec![1, 4, 6, 9]
        );

        matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&[0, 1, 2], &edges), true);
        matrix.add_constraint(vertices[1].downgrade(), &edge_vec_from_indices(&[1, 3], &edges), false);
        matrix.add_constraint(vertices[2].downgrade(), &edge_vec_from_indices(&[0, 3], &edges), true);

        // Test Swap
        matrix.swap_row(2, 1);
        assert_eq!(matrix.get_rhs(1), true); // Row 2 was true, now at index 1
        
        // Test XOR
        matrix.xor_row(0, 1);
        // Column 0: Row 0 (1) XOR Row 1 (1) = 0
        assert_eq!(matrix.get_lhs(0, 0), false);
    }

    #[test]
    fn basic_matrix_manual_echelon() {
        let mut matrix = BasicMatrix::new();
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(
            vec![0, 1, 2], 
            vec![1, 4, 6, 9]
        );

        matrix.add_constraint(vertices[0].downgrade(), &edge_vec_from_indices(&[0, 1, 2], &edges), true);
        matrix.add_constraint(vertices[1].downgrade(), &edge_vec_from_indices(&[1, 3], &edges), false);
        matrix.add_constraint(vertices[2].downgrade(), &edge_vec_from_indices(&[0, 3], &edges), true);

        matrix.xor_row(2, 0);
        matrix.xor_row(0, 1);
        matrix.xor_row(2, 1);
        matrix.xor_row(0, 2);

        // Verify specific cell after operations (Row 0, Var 0 should be 1)
        assert_eq!(matrix.get_lhs(0, 0), true);
        // Row 2, Var 2 (edge 6) should be 1
        assert_eq!(matrix.get_lhs(2, 2), true);
    }
}