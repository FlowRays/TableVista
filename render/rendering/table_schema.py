from dataclasses import dataclass
from typing import List


@dataclass
class Cell:
    content: str
    start_row: int
    end_row: int
    start_col: int
    end_col: int
    is_header: bool = False

    @property
    def rowspan(self) -> int:
        return self.end_row - self.start_row + 1

    @property
    def colspan(self) -> int:
        return self.end_col - self.start_col + 1


@dataclass
class TableMetadata:
    source: str = ""
    caption: str = ""
    source_format: str = ""


@dataclass
class TableStructure:
    num_rows: int
    num_cols: int
    cells: List[Cell]


@dataclass
class CanonicalTable:
    metadata: TableMetadata
    structure: TableStructure


def create_simple_table(
    data: List[List[str]], has_header: bool = True
) -> CanonicalTable:
    if not data:
        raise ValueError("Data cannot be empty")

    num_rows = len(data)
    num_cols = len(data[0])

    cells = []
    for row_idx, row in enumerate(data):
        for col_idx, content in enumerate(row):
            cell = Cell(
                content=str(content),
                start_row=row_idx,
                end_row=row_idx,
                start_col=col_idx,
                end_col=col_idx,
                is_header=(has_header and row_idx == 0),
            )
            cells.append(cell)

    metadata = TableMetadata()
    structure = TableStructure(num_rows=num_rows, num_cols=num_cols, cells=cells)

    return CanonicalTable(metadata=metadata, structure=structure)
