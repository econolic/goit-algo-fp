"""
Pythagorean Tree Fractal Generator
==================================

Generates and visualizes Pythagorean tree fractals with various styling options.

CLI Usage:
    python pythagorean_tree.py --depth 5 --angle 45 --style default --color plasma --output tree.png
    python pythagorean_tree.py --depth 4 --style skeleton --color maroon
    python pythagorean_tree.py --depth 8 --style centerline --color viridis --figure-size 12.0

"""
import argparse
import math
import time
from functools import wraps
from typing import List, Tuple, Optional, Union, Dict, Any, cast

import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.colors import Colormap

# --- Utility Classes and Functions ---

class Point:
    """Represents a 2D point with x and y coordinates."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        return f"Point({self.x:.2f}, {self.y:.2f})"

    def distance_to(self, other: 'Point') -> float:
        """Calculates Euclidean distance to another point."""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

    def rotate_around(self, center: 'Point', angle_rad: float) -> 'Point':
        """Rotates the point around a center by a given angle in radians."""
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        dx = self.x - center.x
        dy = self.y - center.y
        new_x = dx * cos_a - dy * sin_a
        new_y = dx * sin_a + dy * cos_a
        return Point(new_x + center.x, new_y + center.y)


class Square:
    """Represents a square defined by its four corner points."""

    def __init__(self, bottom_left: Point, bottom_right: Point,
                 top_right: Point, top_left: Point) -> None:
        self.bottom_left = bottom_left
        self.bottom_right = bottom_right
        self.top_right = top_right
        self.top_left = top_left

    def get_vertices(self) -> List[Tuple[float, float]]:
        """Returns the square's vertices as a list of (x, y) tuples."""
        return [
            (self.bottom_left.x, self.bottom_left.y),
            (self.bottom_right.x, self.bottom_right.y),
            (self.top_right.x, self.top_right.y),
            (self.top_left.x, self.top_left.y)
        ]

    def side_length(self) -> float:
        """Calculates the side length of the square."""
        return self.bottom_left.distance_to(self.bottom_right)

    def get_top_edge(self) -> Tuple[Point, Point]:
        """Returns the top-left and top-right points forming the top edge."""
        return self.top_left, self.top_right

    def get_bottom_edge(self) -> Tuple[Point, Point]:
        """Returns the bottom-left and bottom-right points."""
        return self.bottom_left, self.bottom_right


def performance_timer(func):
    """Decorator to measure and print function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"INFO: {func.__name__} executed in {execution_time:.6f} s")
        return result
    return wrapper

# --- Pythagorean Tree Generation Logic ---

class PythagoreanTree:
    """
    Generates the geometric elements for a Pythagorean tree fractal.
    """
    DEFAULT_COLORMAP = 'plasma'
    DEFAULT_SINGLE_COLOR = '#800000' # Maroon
    DEFAULT_EDGE_COLOR = '#303030'   # Dark Gray

    def __init__(self, angle_degrees: float = 45.0) -> None:
        if not (0 < angle_degrees < 90):
            raise ValueError("Angle must be between 0 and 90 degrees.")
        self.angle_degrees = angle_degrees
        self.angle_radians = math.radians(angle_degrees)

        # Stores (Square, depth_level)
        self.elements: List[Tuple[Square, int]] = []
        # Stores (Point, Point) for centerline style
        self.centerline_segments: List[Tuple[Point, Point]] = []
        # Stores (apex_point, depth_level) for twig bases in skeleton style
        self.twig_bases: List[Tuple[Point, int, Square]] = []
        # Counters for statistics
        self.generation_count = 0
        self.max_built_depth = 0


    @performance_timer
    def build_fractal(self, target_max_depth: int, base_square_size: float = 100.0) -> None:
        """
        Builds the fractal elements up to the target maximum depth.
        Populates self.elements and self.centerline_segments.
        """
        self.elements = []
        self.centerline_segments = []
        self.twig_bases = []
        self.generation_count = 0
        self.max_built_depth = target_max_depth

        base_sq = self._create_base_square(base_square_size)
        self._add_element(base_sq, 0)

        # Initial "trunk" for centerline style
        m_bottom_base = Point(
            (base_sq.bottom_left.x + base_sq.bottom_right.x) / 2,
            (base_sq.bottom_left.y + base_sq.bottom_right.y) / 2
        )
        m_top_base = Point(
            (base_sq.top_left.x + base_sq.top_right.x) / 2,
            (base_sq.top_left.y + base_sq.top_right.y) / 2
        )
        self.centerline_segments.append((m_bottom_base, m_top_base)) # Spine of base

        apex_of_base = self._calculate_triangle_apex(base_sq.top_left, base_sq.top_right)
        self.centerline_segments.append((m_top_base, apex_of_base)) # To first branch point
        self.twig_bases.append((apex_of_base, 0, base_sq))


        if target_max_depth > 0:
            self._generate_recursive(
                parent_square=base_sq,
                current_depth=0,
                connection_point_from_parent_apex=apex_of_base
            )

        print(f"INFO: Generated {self.generation_count} square elements up to depth {target_max_depth}.")
        if self.centerline_segments:
             print(f"INFO: Generated {len(self.centerline_segments)} centerline segments.")


    def _add_element(self, square: Square, depth: int) -> None:
        """Adds a square element and increments count."""
        self.elements.append((square, depth))
        self.generation_count += 1

    def _create_base_square(self, size: float, center_x: float = 0.0,
                           center_y: float = -50.0) -> Square:
        """Creates the initial square at the base of the tree."""
        half_size = size / 2
        return Square(
            Point(center_x - half_size, center_y - half_size),
            Point(center_x + half_size, center_y - half_size),
            Point(center_x + half_size, center_y + half_size),
            Point(center_x - half_size, center_y + half_size)
        )

    def _calculate_triangle_apex(self, p1: Point, p2: Point) -> Point:
        """
        Calculates the apex of the right-angled triangle formed on the edge (p1, p2).
        p1-p2 is the hypotenuse. The right angle is at the apex.
        self.angle_radians is one of the acute angles (e.g., angle at p1).
        """
        vec_x = p2.x - p1.x
        vec_y = p2.y - p1.y
        base_len = math.sqrt(vec_x**2 + vec_y**2)

        if base_len < 1e-9: return p1 # Avoid division by zero for tiny edges

        # Angle of the base vector p1->p2
        alpha_base = math.atan2(vec_y, vec_x)
        # Length of side adjacent to p1 (p1 to apex)
        len_adj = base_len * math.cos(self.angle_radians)
        # Angle of side p1-apex
        angle_adj = alpha_base + self.angle_radians # Outward construction

        apex_x = p1.x + len_adj * math.cos(angle_adj)
        apex_y = p1.y + len_adj * math.sin(angle_adj)
        return Point(apex_x, apex_y)

    def _create_square_from_edge(self, p1: Point, p2: Point) -> Square:
        """Creates a square where p1-p2 is one edge (e.g., bottom-left to bottom-right)."""
        edge_x = p2.x - p1.x
        edge_y = p2.y - p1.y
        # Perpendicular vector (rotated +90 deg) to build "upwards"
        perp_x = -edge_y
        perp_y = edge_x
        p3 = Point(p2.x + perp_x, p2.y + perp_y) # Top-right
        p4 = Point(p1.x + perp_x, p1.y + perp_y) # Top-left
        return Square(p1, p2, p3, p4)

    def _generate_child_squares(self, parent_square: Square) -> Tuple[Square, Square]:
        """Generates the two child squares branching from the parent square."""
        top_left, top_right = parent_square.get_top_edge()
        apex = self._calculate_triangle_apex(top_left, top_right)
        
        # Order of points for _create_square_from_edge matters for orientation.
        # Left square's base is (top_left, apex)
        left_sq = self._create_square_from_edge(top_left, apex)
        # Right square's base is (apex, top_right)
        right_sq = self._create_square_from_edge(apex, top_right)
        return left_sq, right_sq

    def _generate_recursive(self, parent_square: Square, current_depth: int,
                            connection_point_from_parent_apex: Point) -> None:
        """
        Recursive helper for building fractal elements and centerlines.
        """
        next_depth = current_depth + 1
        if next_depth > self.max_built_depth:
            return

        left_sq, right_sq = self._generate_child_squares(parent_square)

        # Process Left Child
        self._add_element(left_sq, next_depth)
        apex_l = self._calculate_triangle_apex(left_sq.top_left, left_sq.top_right)
        self.centerline_segments.append((connection_point_from_parent_apex, apex_l))
        self.twig_bases.append((apex_l, next_depth, left_sq))

        if next_depth < self.max_built_depth:
            self._generate_recursive(left_sq, next_depth, apex_l)

        # Process Right Child
        self._add_element(right_sq, next_depth)
        apex_r = self._calculate_triangle_apex(right_sq.top_left, right_sq.top_right)
        self.centerline_segments.append((connection_point_from_parent_apex, apex_r))
        self.twig_bases.append((apex_r, next_depth, right_sq))


        if next_depth < self.max_built_depth:
            self._generate_recursive(right_sq, next_depth, apex_r)


    @performance_timer
    def visualize(self, args: argparse.Namespace) -> None:
        """
        Visualizes the generated fractal based on command-line arguments.
        """
        if not self.elements:
            print("WARNING: No elements to visualize. Build the fractal first.")
            return

        fig, ax = plt.subplots(figsize=(args.figure_size, args.figure_size))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        ax.set_aspect('equal')

        # Determine color palette
        is_gradient = False
        color_palette: Union[str, Colormap] # Add type hint for clarity
        try:
            # Check if it's a known colormap
            color_palette = plt.get_cmap(args.color) # Replace plt.cm.get_cmap with plt.get_cmap
            is_gradient = True
            print(f"INFO: Using gradient colormap: {args.color}")
        except ValueError:
            # Assume it's a single color string (hex or name)
            try:
                mcolors.to_rgba(args.color) # Validate color
                color_palette = args.color
                is_gradient = False
                print(f"INFO: Using single color: {args.color}")
            except ValueError:
                print(f"WARNING: Invalid color/colormap '{args.color}'. Defaulting to single color '{self.DEFAULT_SINGLE_COLOR}'.")
                color_palette = self.DEFAULT_SINGLE_COLOR
                is_gradient = False

        # Common plot styling for cleaner views
        if args.style in ['skeleton', 'centerline']:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
        else: # Default style
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.set_xlabel('X coordinate')
            ax.set_ylabel('Y coordinate')

        # --- Style-specific drawing ---
        if args.style == 'default':
            self._draw_default_style(ax, color_palette, is_gradient, args)
        elif args.style == 'skeleton':
            self._draw_skeleton_style(ax, color_palette, is_gradient, args)
        elif args.style == 'centerline':
            self._draw_centerline_style(ax, color_palette, is_gradient, args)

        # Final plot adjustments
        ax.autoscale_view() # Fit elements tightly
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_padding = (xlim[1] - xlim[0]) * 0.05
        y_padding = (ylim[1] - ylim[0]) * 0.05
        ax.set_xlim(xlim[0] - x_padding, xlim[1] + x_padding)
        ax.set_ylim(ylim[0] - y_padding, ylim[1] + y_padding)

        # Title
        plot_title = self._generate_plot_title(args)
        ax.set_title(plot_title, fontsize=14, fontweight='bold', pad=15)

        plt.tight_layout(rect=(0, 0, 1, 0.96)) # Adjust for suptitle
        if args.output:
            fig_face_color = 'white' if args.style in ['skeleton', 'centerline'] else fig.get_facecolor()
            plt.savefig(args.output, dpi=300, bbox_inches='tight', facecolor=fig_face_color)
            print(f"INFO: Figure saved to {args.output}")
        if not args.headless:
            plt.show()
        else:
            plt.close(fig)

    def _generate_plot_title(self, args: argparse.Namespace) -> str:
        """Generates a dynamic plot title."""
        style_map = {
            'default': 'Classic',
            'skeleton': 'Skeleton & Twigs',
            'centerline': 'Centerline Connections'
        }
        title = f"Pythagorean Tree ({style_map.get(args.style, args.style.capitalize())} Style)\n"
        title += f"Depth: {self.max_built_depth}, Angle: {self.angle_degrees}°"
        color_info = args.color if isinstance(args.color, str) else args.color.name
        title += f", Color: {color_info.capitalize()}"
        return title

    def _get_color_for_element(self, base_color_palette: Union[str, mcolors.Colormap],
                               is_gradient: bool, index: int, total_items: int,
                               depth: Optional[int] = None, max_depth: Optional[int] = None) -> Union[str, Tuple[float, float, float, float]]:
        """Gets color for an element based on palette type and index/depth."""
        if not is_gradient:
            # If not a gradient, base_color_palette is the single color string
            # Return the string directly
            return cast(str, base_color_palette) # Ensure it's treated as str

        # Gradient logic: base_color_palette is a Colormap
        # Ensure it's a Colormap before calling it
        if isinstance(base_color_palette, mcolors.Colormap):
            if total_items <= 1:
                 return base_color_palette(0.5) # Return an RGBA tuple from the colormap

            # Option 1: Color by generation index (default for now)
            norm_value = index / (total_items -1) if total_items >1 else 0.5

            # Option 2: Color by depth (if depth info is available and preferred)
            if depth is not None and max_depth is not None and max_depth > 0:
                 norm_value = depth / max_depth

            return base_color_palette(norm_value) # Return an RGBA tuple
        else:
            # Fallback if is_gradient is True but palette is not a Colormap (shouldn't happen with correct logic)
            print("WARNING: Expected Colormap but got something else when is_gradient is True.")
            return (0.5, 0.5, 0.5, 1.0) # Fallback grey
    def _draw_default_style(self, ax: Axes, C_PALETTE: Union[str, mcolors.Colormap],
                           IS_GRADIENT: bool, args: argparse.Namespace) -> None:
        """Draws the default style (filled squares, optional triangles)."""
        num_elements = len(self.elements)
        edge_color_val = self.DEFAULT_EDGE_COLOR if args.show_square_edges else 'none'

        # Draw squares
        for i, (square, depth) in enumerate(self.elements):
            color = self._get_color_for_element(C_PALETTE, IS_GRADIENT, i, num_elements, depth, self.max_built_depth)
            poly = patches.Polygon(square.get_vertices(), closed=True,
                                   facecolor=color, edgecolor=edge_color_val,
                                   linewidth=0.75, alpha=0.85)
            ax.add_patch(poly)

        # Draw construction triangles (if enabled)
        if args.show_triangles:
            for i, (square, depth) in enumerate(self.elements):
                if depth < self.max_built_depth:
                    # Determine triangle color based on parent square's color logic
                    base_sq_color = self._get_color_for_element(C_PALETTE, IS_GRADIENT, i, num_elements, depth, self.max_built_depth)

                    if IS_GRADIENT and isinstance(C_PALETTE, mcolors.Colormap):
                        # For gradient, use a slightly offset color from the colormap
                        norm_val = (i + 0.5) / num_elements if num_elements > 0 else 0.5
                        tri_facecolor = C_PALETTE(np.clip(norm_val, 0, 1))
                        tri_edgecolor = mcolors.to_rgba(tri_facecolor, alpha=0.7)
                    else:
                        # For single color, use the same color but with reduced alpha for face
                        try:
                            rgba = list(mcolors.to_rgba(base_sq_color))
                            tri_facecolor = tuple(rgba[:3] + [rgba[3] * 0.5]) # Reduce alpha
                            tri_edgecolor = mcolors.to_rgba(base_sq_color, alpha=0.7) # Darker edge
                        except:
                            tri_facecolor = (0.7, 0.7, 0.7, 0.4) # Fallback grey transparent face
                            tri_edgecolor = (0.3, 0.3, 0.3, 0.7) # Fallback dark grey transparent edge

                    top_l, top_r = square.get_top_edge()
                    apex = self._calculate_triangle_apex(top_l, top_r)
                    tri_verts = [(top_l.x, top_l.y), (top_r.x, top_r.y), (apex.x, apex.y)]
                    triangle = patches.Polygon(tri_verts, closed=True,
                                               facecolor=tri_facecolor,
                                               edgecolor=tri_edgecolor,
                                               linewidth=0.5, alpha=0.4) # Triangles more transparent
                    ax.add_patch(triangle)

        if IS_GRADIENT and num_elements > 0:
            sm = plt.cm.ScalarMappable(cmap=C_PALETTE, norm=Normalize(vmin=0, vmax=self.max_built_depth)) # Color by depth
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.75, aspect=20, label="Depth Level / Generation Order")


    def _draw_skeleton_style(self, ax: Axes, C_PALETTE: Union[str, mcolors.Colormap],
                            IS_GRADIENT: bool, args: argparse.Namespace) -> None:
        """Draws the skeleton style (square outlines and twigs)."""
        num_elements_for_color = len(self.elements) + len(self.twig_bases) # Rough total for gradient span
        
        # Draw square outlines
        for i, (square, depth) in enumerate(self.elements):
            color = self._get_color_for_element(C_PALETTE, IS_GRADIENT, i, num_elements_for_color, depth, self.max_built_depth)
            poly = patches.Polygon(square.get_vertices(), closed=True,
                                   facecolor='none', edgecolor=color,
                                   linewidth=1.0)
            ax.add_patch(poly)

        # Draw twigs
        twig_len_factor = 0.3
        twig_angle_offset_rad = math.radians(30)
        
        twig_color_start_index = len(self.elements) # For gradient continuity
        
        for i, (base_pt, depth, parent_sq) in enumerate(self.twig_bases):
            # Color twigs based on their parent square's depth or a continuous index
            color = self._get_color_for_element(C_PALETTE, IS_GRADIENT, twig_color_start_index + i,
                                                num_elements_for_color, depth, self.max_built_depth)

            # Twigs emanate from the apex ('base_pt') of the parent_sq
            # Direction vector: from midpoint of parent_sq's top edge towards base_pt (apex)
            # This is slightly simplified; true direction is normal to parent_sq's top edge at apex.
            # For simplicity, let's make twigs relative to the apex (base_pt) and parent square size.
            
            top_l, top_r = parent_sq.get_top_edge()
            mid_top_parent = Point((top_l.x + top_r.x)/2, (top_l.y + top_r.y)/2)

            dir_x = base_pt.x - mid_top_parent.x
            dir_y = base_pt.y - mid_top_parent.y
            norm = math.sqrt(dir_x**2 + dir_y**2)
            
            if norm < 1e-9: udx, udy = 0,1 # Default upwards if no clear direction
            else: udx, udy = dir_x / norm, dir_y / norm

            twig_len = parent_sq.side_length() * twig_len_factor
            if twig_len < 1e-9: twig_len = 0.5 # Min length for tiny squares

            # Central twig
            p_end_mid = Point(base_pt.x + udx * twig_len, base_pt.y + udy * twig_len)
            ax.add_line(mlines.Line2D([base_pt.x, p_end_mid.x], [base_pt.y, p_end_mid.y], color=color, linewidth=0.8))

            # Side twigs (rotated from central direction)
            for angle_offset in [-twig_angle_offset_rad, twig_angle_offset_rad]:
                cos_o, sin_o = math.cos(angle_offset), math.sin(angle_offset)
                udx_rot = udx * cos_o - udy * sin_o
                udy_rot = udx * sin_o + udy * cos_o
                p_end_side = Point(base_pt.x + udx_rot * twig_len, base_pt.y + udy_rot * twig_len)
                ax.add_line(mlines.Line2D([base_pt.x, p_end_side.x], [base_pt.y, p_end_side.y], color=color, linewidth=0.8))


    def _draw_centerline_style(self, ax: Axes, C_PALETTE: Union[str, mcolors.Colormap],
                              IS_GRADIENT: bool, args: argparse.Namespace) -> None:
        """Draws the centerline style (connected apex-to-apex lines)."""
        if not self.centerline_segments:
            print("WARNING: No centerline segments generated for centerline style.")
            return
        
        num_segments = len(self.centerline_segments)
        for i, (p1, p2) in enumerate(self.centerline_segments):
            # For centerline, gradient by segment index is straightforward
            color = self._get_color_for_element(C_PALETTE, IS_GRADIENT, i, num_segments)
            ax.add_line(mlines.Line2D([p1.x, p2.x], [p1.y, p2.y],
                                      color=color, linewidth=1.2))

# --- Argument Parsing and Main Execution ---

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pythagorean Tree Fractal Generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Core parameters
    parser.add_argument(
        "-d", "--depth", type=int, default=5,
        help="Recursion depth of the fractal (0-indexed)."
    )
    parser.add_argument(
        "-a", "--angle", type=float, default=45.0,
        help="Branching angle in degrees for triangle construction."
    )
    parser.add_argument(
        "--base-size", type=float, default=80.0,
        help="Side length of the base square."
    )
    # Visualization style
    parser.add_argument(
        "--style", choices=['default', 'skeleton', 'centerline'], default='default',
        help="Visualization style."
    )
    # Coloring
    parser.add_argument(
        "--color", type=str, default=PythagoreanTree.DEFAULT_COLORMAP,
        help="Color for drawing: can be a Matplotlib colormap name (for gradient) "
             "or a single color string (e.g., 'maroon', '#FF0000')."
    )
    # Options for 'default' style
    parser.add_argument(
        "--show-triangles", action=argparse.BooleanOptionalAction, default=True,
        help="Show construction triangles (for 'default' style)."
    )
    parser.add_argument(
        "--show-square-edges", action=argparse.BooleanOptionalAction, default=True,
        help="Show square edges (for 'default' style)."
    )
    # Output and control
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save plot to specified file path (e.g., tree.png)."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Do not display the plot interactively."
    )
    parser.add_argument(
        "--figure-size", type=float, default=10.0,
        help="Size of the plot figure (width and height in inches)."
    )
    # Testing and profiling
    parser.add_argument("--test", action="store_true", help="Run internal tests and exit.")
    parser.add_argument("--profile", action="store_true", help="Run complexity analysis and exit.")

    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    """Validates specific command-line arguments."""
    if args.depth < 0:
        print("ERROR: Depth must be non-negative.")
        exit(1)
    # Angle validation is handled in PythagoreanTree constructor now.
    if args.figure_size <= 0:
        print("ERROR: Figure size must be positive.")
        exit(1)


def run_tests() -> None:
    """Runs internal consistency tests."""
    print("\nINFO: --- Running Internal Tests ---")
    # Point Class Tests
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    assert abs(p1.distance_to(p2) - 5.0) < 1e-9, "Point distance failed"
    p_rot = Point(1,0).rotate_around(Point(0,0), math.pi/2)
    assert abs(p_rot.x - 0) < 1e-9 and abs(p_rot.y - 1) < 1e-9, "Point rotation failed"
    print("INFO: Point class tests passed.")

    # Square Class Tests
    sq = Square(Point(0,0), Point(1,0), Point(1,1), Point(0,1))
    assert abs(sq.side_length() - 1.0) < 1e-9, "Square side length failed"
    assert sq.get_top_edge()[0].y == 1 and sq.get_top_edge()[1].y == 1, "Square top edge failed"
    print("INFO: Square class tests passed.")

    # PythagoreanTree Basic Build
    try:
        tree_test = PythagoreanTree(angle_degrees=30.0)
        tree_test.build_fractal(target_max_depth=2, base_square_size=10)
        # Expected: 1 (base) + 2 (d1) + 4 (d2) = 7 elements
        assert tree_test.generation_count == 7, f"Element count mismatch: {tree_test.generation_count}"
        assert tree_test.max_built_depth == 2, "Max depth mismatch"
        print("INFO: PythagoreanTree build basics passed.")
        
        # Apex calculation check (e.g., 45 degrees)
        tree_45 = PythagoreanTree(angle_degrees=45.0)
        apex_45 = tree_45._calculate_triangle_apex(Point(0,0), Point(10,0))
        # For 45 deg, apex should be (5,5) if hypotenuse is (0,0)-(10,0)
        assert abs(apex_45.x - 5.0) < 1e-9 and abs(apex_45.y - 5.0) < 1e-9, "45-deg apex calculation failed"
        print("INFO: PythagoreanTree apex calculation (45-deg) passed.")

    except Exception as e:
        print(f"ERROR: Test failed: {e}")
        exit(1)
    print("INFO: --- All Internal Tests Passed Successfully ---")


def run_complexity_analysis(max_depth_for_analysis: int) -> None:
    """Runs and plots a basic complexity analysis."""
    print("\nINFO: --- Running Complexity Analysis ---")
    tree_analyzer = PythagoreanTree(angle_degrees=45.0) # Fixed angle for analysis
    depths = list(range(max_depth_for_analysis + 1))
    times = []
    element_counts = []

    for depth in depths:
        print(f"INFO: Analyzing depth {depth}...")
        start_t = time.perf_counter()
        tree_analyzer.build_fractal(target_max_depth=depth)
        end_t = time.perf_counter()
        times.append(end_t - start_t)
        element_counts.append(tree_analyzer.generation_count)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Complexity Analysis of Pythagorean Tree Generation", fontsize=16)

    ax1.semilogy(depths, times, 'bo-', label="Execution Time")
    ax1.set_xlabel("Recursion Depth")
    ax1.set_ylabel("Time (seconds, log scale)")
    ax1.set_title("Time Complexity")
    ax1.grid(True, linestyle=':')
    ax1.legend()

    ax2.plot(depths, element_counts, 'ro-', label="Actual Elements")
    theoretical_elements = [2**(d + 1) - 1 for d in depths]
    ax2.plot(depths, theoretical_elements, 'g--', label="Theoretical Max (2^(d+1)-1)")
    ax2.set_xlabel("Recursion Depth")
    ax2.set_ylabel("Number of Squares")
    ax2.set_title("Space Complexity (Elements)")
    ax2.legend()
    ax2.grid(True, linestyle=':')
    try: # Use semilogy if values are large enough, otherwise linear
      if max(element_counts) > 100: ax2.set_yscale('log')
    except ValueError: pass


    plt.tight_layout(rect=(0, 0, 1, 0.96)) # Adjust for suptitle
    plt.show()
    print("INFO: --- Complexity Analysis Complete ---")


def main() -> None:
    """Main function to generate and visualize the Pythagorean tree."""
    args = parse_arguments()

    if args.test:
        run_tests()
        return

    if args.profile:
        run_complexity_analysis(args.depth)
        return

    validate_arguments(args)

    print(f"INFO: Generating Pythagorean tree: Depth={args.depth}, Angle={args.angle}, Style='{args.style}'")
    try:
        tree = PythagoreanTree(angle_degrees=args.angle)
    except ValueError as e:
        print(f"ERROR: {e}")
        exit(1)
        
    tree.build_fractal(target_max_depth=args.depth, base_square_size=args.base_size)

    if args.output or not args.headless:
        tree.visualize(args)
    else:
        print("INFO: Headless mode: Plot generation skipped.")

if __name__ == "__main__":
    main()