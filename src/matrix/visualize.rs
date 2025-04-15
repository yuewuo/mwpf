//! Parity Matrix Table
//!
//! Converting a matrix into a printable table.
//!
//! I created my own Table struct as a simple wrapper on prettytable::Table
//! because it doesn't provide any public method to retrieve the title row.
//! Some of my functionalities require a flexible operation on the title row.
//!

use super::interface::*;
#[cfg(feature = "unsafe_pointer")]
use crate::pointers::UnsafePtr;
use crate::util::*;
use prettytable::format::TableFormat;
use prettytable::*;

#[derive(Clone)]
pub struct VizTable {
    pub title: Row,
    pub rows: Vec<Row>,
    pub edges: Vec<EdgeIndex>,
}

impl VizTable {
    pub fn force_single_column(long_str: &str) -> String {
        long_str
            .chars()
            .enumerate()
            .flat_map(|(idx, c)| if idx == 0 { vec![c] } else { vec!['\n', c] })
            .collect()
    }
}

lazy_static! {
    pub static ref DEFAULT_TABLE_FORMAT: TableFormat = {
        let mut format = TableFormat::new();
        format.padding(0, 0);
        format.column_separator('\u{250A}');
        format.borders('\u{250A}');
        use format::LinePosition::*;
        let separators = [
            (Intern, ['\u{2500}', '\u{253C}', '\u{251C}', '\u{2524}']),
            (Top, ['\u{2500}', '\u{252C}', '\u{250C}', '\u{2510}']),
            (Bottom, ['\u{2500}', '\u{2534}', '\u{2514}', '\u{2518}']),
            (Title, ['\u{2550}', '\u{256A}', '\u{255E}', '\u{2561}']),
        ];
        for (position, s) in separators {
            format.separators(&[position], format::LineSeparator::new(s[0], s[1], s[2], s[3]))
        }
        format
    };
}

impl<M: MatrixView> From<&mut M> for VizTable {
    fn from(matrix: &mut M) -> VizTable {
        // create title
        let mut title = Row::empty();
        title.add_cell(Cell::new(""));
        let mut edges = vec![];
        for column in 0..matrix.columns() {
            let var_index = matrix.column_to_var_index(column);
            let edge_weak = matrix.var_to_edge_weak(var_index);
            let edge_index = edge_weak.upgrade_force().read_recursive().edge_index;
            edges.push(edge_index);
            let edge_index_str = Self::force_single_column(edge_index.to_string().as_str());
            title.add_cell(Cell::new(edge_index_str.as_str()).style_spec("brFm"));
        }
        title.add_cell(Cell::new(" = "));
        // create body rows
        let mut rows: Vec<Row> = vec![];
        for row in 0..matrix.rows() {
            let mut table_row = Row::empty();
            table_row.add_cell(Cell::new(row.to_string().as_str()).style_spec("brFb"));
            for column in 0..matrix.columns() {
                let var_index = matrix.column_to_var_index(column);
                table_row.add_cell(Cell::new(if matrix.get_lhs(row, var_index) { "1" } else { " " }));
            }
            table_row.add_cell(Cell::new(if matrix.get_rhs(row) { " 1 " } else { "   " }));
            rows.push(table_row);
        }
        VizTable { title, rows, edges }
    }
}

impl From<VizTable> for Table {
    fn from(viz_table: VizTable) -> Table {
        let mut table = Table::new();
        table.set_format(*DEFAULT_TABLE_FORMAT);
        table.set_titles(viz_table.title.clone());
        for row in viz_table.rows.iter() {
            table.add_row(row.clone());
        }
        table
    }
}

impl From<VizTable> for serde_json::Value {
    fn from(viz_table: VizTable) -> serde_json::Value {
        let mut table_json = vec![];
        let mut title_json = vec![];
        for cell in viz_table.title.iter() {
            title_json.push(cell.get_content());
        }
        table_json.push(title_json);
        for row in viz_table.rows.iter() {
            let mut row_json = vec![];
            for cell in row {
                row_json.push(cell.get_content());
            }
            table_json.push(row_json);
        }
        json!(table_json)
    }
}

pub trait VizTrait {
    fn viz_table(&mut self) -> VizTable;
    fn printstd_str(&mut self) -> String {
        Table::from(self.viz_table()).to_string().replace("\r", "")
    }
    fn printstd(&mut self) {
        #[cfg(feature = "colorful")]
        Table::from(self.viz_table()).printstd();
        #[cfg(not(feature = "colorful"))]
        println!("{}", Table::from(self.viz_table()));
    }
}

impl VizTrait for VizTable {
    fn viz_table(&mut self) -> VizTable {
        self.clone()
    }
}

impl VizTable {
    pub fn snapshot(&self) -> serde_json::Value {
        json!({
            "version": env!("CARGO_PKG_VERSION"),
            "table": serde_json::Value::from(self.clone()),
            "edges": self.edges,
        })
    }
}

#[cfg(test)]
pub mod tests {
    use super::super::*;
    use crate::dual_module_pq::{EdgePtr, VertexPtr};
    use crate::matrix::basic::tests::{edge_vec_from_indices, initialize_vertex_edges_for_matrix_testing};
    use std::collections::HashSet;

    #[test]
    fn viz_table_1() {
        // cargo test --features=colorful viz_table_1 -- --nocapture
        let mut matrix = BasicMatrix::new();
        let vertex_indices = vec![0, 1, 2];
        let edge_indices = vec![1, 4, 16, 23];
        let vertex_incident_edges_vec = vec![vec![0, 1, 2], vec![1, 3], vec![0, 3]];
        let (vertices, edges) = initialize_vertex_edges_for_matrix_testing(vertex_indices, edge_indices);
        matrix.add_constraint(
            vertices[0].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[0], &edges),
            true,
        );
        matrix.add_constraint(
            vertices[1].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[1], &edges),
            false,
        );
        matrix.add_constraint(
            vertices[2].downgrade(),
            &edge_vec_from_indices(&vertex_incident_edges_vec[2], &edges),
            true,
        );
        matrix.printstd();
        assert_eq!(
            matrix.clone().printstd_str(),
            "\
┌─┬─┬─┬─┬─┬───┐
┊ ┊1┊4┊1┊2┊ = ┊
┊ ┊ ┊ ┊6┊3┊   ┊
╞═╪═╪═╪═╪═╪═══╡
┊0┊1┊1┊1┊ ┊ 1 ┊
├─┼─┼─┼─┼─┼───┤
┊1┊ ┊1┊ ┊1┊   ┊
├─┼─┼─┼─┼─┼───┤
┊2┊1┊ ┊ ┊1┊ 1 ┊
└─┴─┴─┴─┴─┴───┘
"
        );
        let mut viz_table = matrix.viz_table();
        assert_eq!(
            serde_json::Value::from(viz_table.viz_table()),
            json!([
                ["", "1", "4", "1\n6", "2\n3", " = "],
                ["0", "1", "1", "1", " ", " 1 "],
                ["1", " ", "1", " ", "1", "   "],
                ["2", "1", " ", " ", "1", " 1 "]
            ])
        )
    }
}
