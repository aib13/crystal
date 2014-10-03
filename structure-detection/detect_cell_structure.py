from collections import defaultdict, namedtuple
from itertools import izip
from magic_iterators import indices_of, intersection, pairwise, renumber_keys, unique_values



def _intersect(a, b):
# Gives the intersection of 2 lists

    return list(set(a) & set(b))


def _contains(small, big):
# Returns is a list is in another list

    for i in small:
        if (not (i in big)):
            return False
    return True


def _difference(a, b):
# Gives the difference of 2 lists

    new_list = []
    for element in a:
        if element not in b:
            new_list.append(element)
    return new_list


def _build_the_cell2cells_map(cell2nodes):
# Builds a map from a cell to its neighbors

    cell2cells = defaultdict(list)

    for cell_id1, nodes1 in enumerate(cell2nodes):
        for cell_id2, nodes2 in enumerate(cell2nodes):
            if (cell_id1 < cell_id2):
                common_nodes = _intersect(nodes1, nodes2)
                if (len(common_nodes) == 2):
                    cell2cells[cell_id1].append(cell_id2)
                    cell2cells[cell_id2].append(cell_id1)

    return cell2cells


def _build_the_node2cells_map(region, cell2nodes):

# Builds a map from a node to the already detected cells
# TO DO: A better idea is to build a cell2edge2node map

    node2cells = defaultdict(list)

    for cell_id in region:

        # Add the cell to the corresponding nodes

        for node in cell2nodes[cell_id]:
            if cell_id not in node2cells[node]:
                node2cells[node].append(cell_id)
        
        # Add the adjacent cells to the corresponding nodes 
        for cell in region[cell_id]:
            
            for node in cell2nodes[cell]:
                if cell not in node2cells[node]:
                    node2cells[node].append(cell)


    return node2cells


def _grow_a_row(cell2cells, cell2nodes, start_cell, node_1, node_2):

# The function gives a row starting with start_cell into the direction given by node_1 and node_2
    
    row = []
    #defaultdict(list)
    
    current_cell = start_cell

    if( current_cell != -1 ):
    
        row.append(start_cell)

        # Extract the current_cell's adjacent list of cells
        list_adjacent_cells = cell2cells[current_cell]

        current_node_1 = node_1
        current_node_2 = node_2

        for num in range(1,10):

            for adjacent_cell in list_adjacent_cells:

                if (adjacent_cell not in row):

                    if (current_node_1 in cell2nodes[adjacent_cell]) and (current_node_2 in cell2nodes[adjacent_cell]): 

                        # We have found an adjacent cell

                        # Append the cell to the detected region
                        row.append(adjacent_cell)
            
                        # Find the common nodes between cell and current_cell
                        common_nodes = _intersect(cell2nodes[adjacent_cell], cell2nodes[current_cell])
            
                        # Find the list of uncommon nodes
                        uncommon_nodes = [ node for node in cell2nodes[adjacent_cell] if node not in common_nodes ]
        
                        # Set the direction, 
                        j = iter(uncommon_nodes)
                        if(len (uncommon_nodes) != 2):
                            return row
                        else:
                            current_node_1 = j.next()
                            current_node_2 = j.next()

                        # Reset the current cell
                        current_cell = adjacent_cell
                        list_adjacent_cells = cell2cells[current_cell]

    return row


def _give_the_south_cell(used_cells, detected_row, cell2cells, cell2nodes, start_cell, node_1, node_2):

# Giving a cell, start_cell, it gives a south cell

    list_adjacent_cells = cell2cells[start_cell]

    unused_nodes = [ node for node in cell2nodes[start_cell] if (node != node_1 and node != node_2) ]

    # Find one of the "north" and "south" cells

    for adjacent_cell in list_adjacent_cells:
        if (adjacent_cell not in detected_row) and (adjacent_cell not in used_cells): 
            if (not _contains(unused_nodes, cell2nodes[adjacent_cell])):
                return adjacent_cell
            
    return -1


def _give_the_south_cells(used_cells, detected_row, cell2cells, cell2nodes, start_cell, node_1, node_2):

# Giving a cell, start_cell, it gives the 2 south cells

    list_adjacent_cells = cell2cells[start_cell]

    unused_nodes = [ node for node in cell2nodes[start_cell] if (node != node_1 and node != node_2) ]

    # Find one of the "north" and "south" cells

    cells = []

    for adjacent_cell in list_adjacent_cells:
        if (adjacent_cell not in detected_row) and (adjacent_cell not in used_cells): 
            if (not _contains(unused_nodes, cell2nodes[adjacent_cell])):
                cells.append(adjacent_cell)
            
    return cells


def _grow_a_row_in_south(used_cells, detected_row, cell2cells, cell2nodes, start_cell, node_1, node_2):

# Giving a cell, start_cell, it gives a south row, pointing into the direction given by node_1 and node_2

    south_row = defaultdict(list)

    # Find one of the "north" and "south" cells
    south_cells = _give_the_south_cells(used_cells, detected_row, cell2cells, cell2nodes, start_cell, node_1, node_2)
         
    for south_cell in south_cells:

        south_row[south_cell].append(south_cell)

        nodes = []

        common_nodes_between_start_cell_and_south_cell = _intersect(cell2nodes[start_cell], cell2nodes[south_cell])

        for node in common_nodes_between_start_cell_and_south_cell:
            if node not in nodes:
                nodes.append(node)

        for node in cell2nodes[south_cell]:
            if node not in nodes:
                nodes.append(node)

        current = south_cell


        for num in range(0, len(detected_row) - 1):

            advance = 0;

            for node in cell2nodes[detected_row[num+1]]:
                if node not in nodes:
                    nodes.append(node)

            for adjacent_cell in cell2cells[current]:
    
                if (len(_intersect(nodes, cell2nodes[adjacent_cell])) == 3) and (adjacent_cell not in south_row[south_cell]) and (adjacent_cell not in detected_row): 

                    # We have found a south adjacent cell

                    # Append the cell to the detected region
                    south_row[south_cell].append(adjacent_cell)
            
                    # Next step
                    current = adjacent_cell

                    for node in cell2nodes[adjacent_cell]:
                        if node not in nodes:
                            nodes.append(node)

                    advance = 1

                    break
            if advance != 1:
                break

    return south_row


def _grow_the_region_in_one_way(cell2nodes, cell2cells, start_cell, node_1, node_2, south_cell, detected_row):

# Gives a full region in one way

    region = defaultdict(list)
    current = start_cell
    used_cells = []

    for cell in detected_row:
        region[current].append(cell)

    num = 1;

    while ( current != -1 ):

        used_cells.append(current)

        if(num != 1):
            south_cell = _give_the_south_cell(used_cells, region[current], cell2cells, cell2nodes, current, node_1, node_2)
        
        num = num + 1

        print "START CELL: ", current
        print "SOUTH CELL: ", south_cell
        print "num: ", num

        print "We are now searching in south into the direction: "
        print "node 1: ", node_1
        print "node 2: ", node_2
        

        if (south_cell not in region[south_cell]):
            if (south_cell != -1):
                region[south_cell].append(south_cell)


        print "south row: ", _grow_a_row_in_south(used_cells, region[current], cell2cells, cell2nodes, current, node_1, node_2)

        south_region = _grow_a_row_in_south(used_cells, region[current], cell2cells, cell2nodes, current, node_1, node_2)

        for cell_id in south_region:
            for cell in south_region[cell_id]:
                if cell not in region[south_cell]:
                    if cell not in region[cell_id]:
                        region[cell_id].append(cell)

        remaining_nodes = _difference(cell2nodes[current], [node_1, node_2])

        print "We are now searching in south into the direction: "
        print "node_3: ", remaining_nodes[0]
        print "node_4: ", remaining_nodes[1]

        print "south row in east: ", _grow_a_row_in_south(used_cells, region[current], cell2cells, cell2nodes, current, remaining_nodes[0], remaining_nodes[1])

        south_region = _grow_a_row_in_south(used_cells, region[current], cell2cells, cell2nodes, current, remaining_nodes[0], remaining_nodes[1])

        for cell_id in south_region:
            for cell in south_region[cell_id]:
                if cell not in region[south_cell]:
                    if cell not in region[cell_id]:
                        region[cell_id].append(cell)


        node2cells = _build_the_node2cells_map(region, cell2nodes)

        print "detected region: ", region

        node_1 = -1
        node_2 = -1

        common_nodes = _intersect(cell2nodes[current], cell2nodes[south_cell])

        for node_id in node2cells: 
            if ( len(node2cells[node_id]) == 4 )  and ( current in node2cells[node_id] )  and ( node_id in common_nodes ) :
                node_1 = node_id
            
        for node_id in node2cells: 
            if ( len(node2cells[node_id]) == 2 )  and ( south_cell in node2cells[node_id] ) and not ( node_id in common_nodes ) :
                node_2 = node_id


        current = south_cell

        used_cells.append(current)
        print "USED NODES: ", used_cells

        print
    
    return region


def _find_a_starting_quad_and_its_region(cell2nodes):

# The main function

    # Find 2 adjacent cells (2 cells that have a common cell)
    # Obtain a key - cell1

    cell2cells = _build_the_cell2cells_map(cell2nodes)

    for start_cell in cell2cells:
        if (len( cell2cells[start_cell] ) != 0) :
            break

    list_adjacent_cells_of_start_cell = cell2cells[start_cell]

    cell2 = list_adjacent_cells_of_start_cell[0]

    region = defaultdict(list)

    # Find the common edge
    common_nodes = _intersect(cell2nodes[start_cell], cell2nodes[cell2])
    structure = _grow_a_row(cell2cells, cell2nodes, start_cell, common_nodes[0], common_nodes[1])
    for cell in structure:
        region[start_cell].append(cell)


    remaining_nodes = _difference (cell2nodes[start_cell], [common_nodes[0], common_nodes[1]])
    structure = _grow_a_row(cell2cells, cell2nodes, start_cell, remaining_nodes[0], remaining_nodes[1])
    for cell in structure:
        region[start_cell].append(cell)
    
    south_cells = _give_the_south_cells([start_cell], region[start_cell], cell2cells, cell2nodes, start_cell, common_nodes[0], common_nodes[1])

    for south_cell in south_cells:

        detected_region = _grow_the_region_in_one_way(cell2nodes, cell2cells, start_cell, common_nodes[0], common_nodes[1], south_cell, region[start_cell])
        for cell_id in detected_region:
            for cell in detected_region[cell_id]:
                if cell not in region[cell_id]:
                    region[cell_id].append(cell)
    
    return region

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# A quad is formed by four nodes, as well as their positions along rows and columns
Quad = namedtuple('Quad', ['row1', 'row2', 'col1', 'col2', 'r1c1', 'r1c2', 'r2c1', 'r2c2', 'first_in_row'])

# IMPORTANT: This defines the compass ordering
Quad.getNodes = lambda self: (self.r1c1, self.r1c2, self.r2c1, self.r2c2)

def _invert_cell2ordnodes(cell2ordnodes):
    node2ordcells = defaultdict(lambda: [-1, -1, -1, -1])

    for cell_id, nodes in enumerate(cell2ordnodes):
        for direction, node_id in enumerate(nodes):
            assert node_id != -1
            assert node2ordcells[node_id][direction] == -1, "Multiple nodes adjacent to the same cell from the same direction: cell %d and node %d" % (cell_id, node_id)
            node2ordcells[node_id][direction] = cell_id

    return node2ordcells

def _print_partial_region(header, region):
    print header
    row_max = min(3, len(region))
    col_max = min(10, len(region[0]))	

    for row in region[:row_max]:
        print row[:col_max], '...'
    print '...'

def _find_a_starting_quad(node_region, node2ordcells, cell2cells):

    # Find the top-left quad
    for quad in _pairwise_enumerate_2d(node_region):
        quad_nodes_to_cells = [ node2ordcells[n] for n in quad.getNodes() ]
        common_cells = intersection(*quad_nodes_to_cells)
        common_cells.discard(-1)

        # No cells in common. Advance to next quad.
        if not common_cells: continue

        # Multiple cells in common - wraparound case
        if len(common_cells) > 1: raise NotImplementedError('Wraparound case')

        # Get the unique cell that is adjacent to all four nodes
        this_cell = iter(common_cells).next()

        # Get all directions that point to this_cell

        matching_directions = [ indices_of(this_cell, cs) for cs in quad_nodes_to_cells ]
        assert all(len(n) == 1 for n in matching_directions), "A node is pointing to the same cell multiple times!"

        # For each node, get its unique direction wrt this_cell
        compass = [ n[0] for n in matching_directions ]
        assert unique_values(compass), "Multiple nodes pointing to the same cell from the same direction!"

        # End search
        return quad, compass

def _find_remaining_structure(topleft_quad, compass, node_region, node2ordcells):
    row_start = topleft_quad.row1
    row_finish = None
    col_start = topleft_quad.col1
    col_finish = None

    current_row = []
    structured_cells = [ current_row ]

    # Traverse first row, from col_start until the last structured column reached
    first2rows = ( node_region[0][col_start:] , node_region[1][col_start:] )
    for quad in _pairwise_enumerate_2d(first2rows):
        # Find pointed-to cells
        pointed_to_cells = set()
        for node, direction in izip(quad.getNodes(), compass):
            pointed_to_cells.add(node2ordcells[node][direction])
        pointed_to_cells.discard(-1)

        # Nodes do not all point to a unique cell - passed last structured column
        if len(pointed_to_cells) != 1:
            col_finish = quad.col1
            break

        # Add cell to structured region
        cell = iter(pointed_to_cells).next()
        current_row.append(cell)

    if not col_finish:
        col_finish = quad.col2

    # Traverse remaining rows, from col_start to col_finish
    second_row_onwards = ( row[col_start:col_finish+1] for row in node_region[1:] )
    for quad in _pairwise_enumerate_2d(second_row_onwards, row_offset=1, col_offset=col_start):
        # If this is the first quad in this row, append a new row to structured_cells
        if quad.first_in_row:
            current_row = []
            structured_cells.append(current_row)

        # Find pointed-to cells
        pointed_to_cells = set(node2ordcells[node][direction]
            for node, direction in izip(quad.getNodes(), compass))

        # Nodes do not all point to a unique cell - remove this row and stop
        if len(pointed_to_cells) != 1:
            structured_cells.pop()
            row_finish = quad.row1
            break

        # Add pointed-to cell to structured region
        cell = iter(pointed_to_cells).next()
        current_row.append(cell)

    if not row_finish:
        print 'finish him'
        row_finish = quad.row2


    num_cell_rows = row_finish - row_start
    num_cell_cols = col_finish - col_start
    num_node_rows = len(node_region)
    num_node_cols = len(node_region[0])

    print 'node region: %d x %d' %(num_node_rows, num_node_cols)
    print 'cell region: %d x %d' %(num_cell_rows, num_cell_cols)
    print 'cell region: rows %d-%d, cols: %d-%d' %(row_start, row_finish, col_start, col_finish)

    assert len(structured_cells) == num_cell_rows, 'Number of rows mismatch'
    assert all(len(row) == num_cell_cols for row in structured_cells), 'Number of cols mismatch'

    cell_region = StructuredCellRegion()
    cell_region.structured_cells = structured_cells
    cell_region.row_start = row_start
    cell_region.row_finish = row_finish
    cell_region.col_start = col_start
    cell_region.col_finish = col_finish
    cell_region.num_cell_rows = num_cell_rows
    cell_region.num_cell_cols = num_cell_cols
    cell_region.compass = compass

    return cell_region


def _renumber_cells(structured_cell_regions, num_cells):
    # Construct reordering for structured cells
    oldcell2newcell = dict()
    newcell2oldcell = []
    visited_structured_cells = set()
    current_offset = 0
    current_new_node_id = 0
    for cell_region in structured_cell_regions:
        cell_region.cells_offset = current_offset
        current_offset += cell_region.num_cell_rows * cell_region.num_cell_cols
        for row in cell_region.structured_cells:
            for old_cell_id in row:
                newcell2oldcell.append(old_cell_id)
                visited_structured_cells.add(old_cell_id)
                oldcell2newcell[old_cell_id] = current_new_node_id
                current_new_node_id += 1

    assert len(visited_structured_cells) == len(newcell2oldcell), "Duplicate structured cells"
    assert len(visited_structured_cells) == current_new_node_id

    # Construct reordering for unstructured cells
    for old_cell_id in xrange(num_cells):
        if old_cell_id not in visited_structured_cells:
            newcell2oldcell.append(old_cell_id)
            oldcell2newcell[old_cell_id] = current_new_node_id
            current_new_node_id += 1

    unstructured_cells_offset = current_offset
    num_structured_cells = len(visited_structured_cells)
    assert len(oldcell2newcell) == len(newcell2oldcell) == num_cells

    return oldcell2newcell, newcell2oldcell, num_structured_cells, unstructured_cells_offset


class StructuredCellRegion:
    pass


class UnstructuredCellRegion:
    pass


class CellStructureFromNodeStructure(object):
    """Given structured node regions, derives structured cell regions"""
    def __init__(self, structured_node_regions, cell2ordnodes):
        num_cells = len(cell2ordnodes)

        node2ordcells = _invert_cell2ordnodes(cell2ordnodes)

        structured_cell_regions = []
        for node_region in structured_node_regions:
            # Preview of node region
            _print_partial_region('Structured node region:', node_region)

            # Extract cell structured region
            topleft_quad, compass = _find_a_starting_quad(node_region, node2ordcells)
            cell_region = _find_remaining_structure(topleft_quad, compass, node_region, node2ordcells)
            structured_cell_regions.append(cell_region)

        # Renumber cells
        oldcell2newcell, newcell2oldcell, num_structured_cells, unstructured_cells_offset = _renumber_cells(structured_cell_regions, num_cells)

        # Determine number of unstructured cells
        assert num_structured_cells == unstructured_cells_offset
        num_unstructured_cells = num_cells - num_structured_cells

        # Unstructured cell region metadata
        unstructured_cell_regions = UnstructuredCellRegion()
        unstructured_cell_regions.unstructured_cells_offset = unstructured_cells_offset
        unstructured_cell_regions.num_unstructured_cells = num_unstructured_cells

        # Apply cell reordering to original map
        new_cell2ordnodes = renumber_keys(cell2ordnodes, newcell2oldcell)

        # Save
        self.new_cell2ordnodes = new_cell2ordnodes
        self.structured_cell_regions = structured_cell_regions
        self.unstructured_cell_regions = unstructured_cell_regions
        self.oldcell2newcell = oldcell2newcell
