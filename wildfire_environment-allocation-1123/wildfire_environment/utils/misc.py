from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from wildfire_environment.core.constants import (
    STATE_IDX_TO_COLOR_WILDFIRE,
    TILE_PIXELS,
)
from wildfire_environment.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_rect,
    point_in_line,
)
from wildfire_environment.core.agent import Agent
from wildfire_environment.core.grid import Grid
from wildfire_environment.core.world import WorldT


def save_frames_as_gif(
    frames, path="./", filename="env-render-episode", ep=0, fps=60, dpi=72
):
    """Save a list of frames as a gif.

    Parameters
    ----------
    frames : ndarray
        list of frames to be saved as gif. Each frame is an RGB image of the environment at a time step.
    path : str, optional
       location to save the gif, by default "./"
    filename : str, optional
        name of the gif, by default "env-render-episode-". The episode number is appended to the filename. Thus, entire path of saved gif is "path/filename"+"ep.gif"
    ep : int, optional
        the episode number during which the frames were recorded, by default 0. For example, it is useful if the frames are saved during RL agent training to identify the episode number during which the frames were recorded.
    fps : int, optional
        frequency at which consecutive images or frames are displayed in the gif, by default 60
    dpi : int, optional
        output resolution of gif, by default 72
    """
    filename = filename + "-" + str(ep) + ".gif"
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=dpi)

    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer="imagemagick", fps=fps)
    plt.close()


def render_activity_gauge(img, agent: Agent, world: WorldT):
    """
    Render activity time gauge above agent.

    Parameters
    ----------
    img : NDArray
        image of wildfire grid
    agent : Agent
        agent to render gauge for
    world : WorldT
        wildfire world

    Returns
    -------
    img : NDArray
        image with gauge rendered
    """
    tile_size = TILE_PIXELS
    pos = agent.pos

    # Calculate gauge fill percentage
    if hasattr(agent, 'max_active_time') and agent.max_active_time > 0:
        fill_ratio = agent.active_time_remaining / agent.max_active_time
    else:
        return img

    # Gauge position: above agent
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 2

    # Background (dark gray)
    img[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]

    # Fill (yellow color for activity time gauge)
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        # Yellow color (RGB: 255, 255, 0)
        color = [255, 255, 0]
        img[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = color

    return img


def render_water_gauge(img, agent: Agent, world: WorldT):
    """
    Render water/suppressant gauge below activity gauge.

    Parameters
    ----------
    img : NDArray
        image of wildfire grid
    agent : Agent
        agent to render gauge for
    world : WorldT
        wildfire world

    Returns
    -------
    img : NDArray
        image with water gauge rendered
    """
    tile_size = TILE_PIXELS
    pos = agent.pos

    # Calculate gauge fill percentage
    if hasattr(agent, 'max_water') and agent.max_water > 0:
        fill_ratio = agent.water_remaining / agent.max_water
    else:
        return img

    # Gauge position: below activity gauge
    gauge_width = int(tile_size * 0.8)
    gauge_height = 3
    gauge_x = pos[0] * tile_size + int(tile_size * 0.1)
    gauge_y = pos[1] * tile_size + 6  # 4 pixels below activity gauge (2+3+1)

    # Background (dark gray)
    img[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+gauge_width] = [50, 50, 50]

    # Fill (cyan/light blue for water)
    fill_width = int(gauge_width * fill_ratio)
    if fill_width > 0:
        # Cyan color (RGB: 0, 255, 255)
        color = [0, 255, 255]
        img[gauge_y:gauge_y+gauge_height, gauge_x:gauge_x+fill_width] = color

    return img


def render_agent_tiles(
    img,
    agent: Agent,
    helper_grid: Grid,
    world: WorldT,
    x_min=None,
    y_min=None,
    x_max=None,
    y_max=None,
    colors=None,
    agent_offset=None,
):
    """
    Re-render the tile containing given agent to add background color corresponding to the state of tree in that cell.

    Parameters
    ----------
    img : NDArray
        image of wildfire grid with tree missing at agent location
    agent : Agent
        agent located in the tile to be re-rendered
    helper_grid : Grid
        grid containing only trees and no agents. Used to get the state of tree in the cell containing the agent
    world : WorldT
        wildfire world
    x_min : list[int], optional
        list of x-coordinates of the left boundary of selfish regions
    y_min : list[int], optional
        list of y-coordinates of the top boundary of selfish regions
    x_max : list[int], optional
        list of x-coordinates of the right boundary of selfish regions
    y_max : list[int], optional
        list of y-coordinates of the bottom boundary of selfish regions
    colors : list, optional
        list of colors of the boundaries of selfish regions. Boundary color is same as the color of the corresponding selfish agent
    agent_offset : tuple[float, float], optional
        offset for rendering agent when multiple agents are at the same position

    Returns
    -------
    img : NDArray
        image with all trees and agents rendered
    """

    pos = agent.pos
    o = helper_grid.get(*pos)
    s = o.state

    tile_size = TILE_PIXELS
    ymin = pos[1] * tile_size
    ymax = (pos[1] + 1) * tile_size
    xmin = pos[0] * tile_size
    xmax = (pos[0] + 1) * tile_size

    tree_color = world.COLORS[STATE_IDX_TO_COLOR_WILDFIRE[s]]

    # Use offset for rendering if provided (for multiple agents on same cell)
    cx, cy = 0.5, 0.5
    if agent_offset is not None:
        cx += agent_offset[0]
        cy += agent_offset[1]

    fill_coords(
        img[ymin:ymax, xmin:xmax, :],
        point_in_circle(cx, cy, 0.25),
        world.COLORS[agent.color],
        bg_color=tree_color,
    )
    i, j = pos
    changed_left_boundary = False
    changed_top_boundary = False
    if colors is not None:
        for index, color in enumerate(colors):
            # check if object is located adjacent to the top boundary of selfish region
            if j == y_min[index]:
                if x_min[index] <= i <= x_max[index]:
                    changed_top_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 1, 0, 0.093),
                        color,
                    )
            # check if object is located adjacent to the left boundary of selfish region
            if i == x_min[index]:
                if y_min[index] <= j <= y_max[index]:
                    changed_left_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 0.093, 0, 1),
                        color,
                    )
            # check if object is located adjacent to the bottom boundary of selfish region
            if j == y_max[index] + 1:
                if x_min[index] <= i <= x_max[index]:
                    changed_top_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 1, 0, 0.093),
                        color,
                    )
            # check if object is located adjacent to the right boundary of selfish region
            if i == x_max[index] + 1:
                if y_min[index] <= j <= y_max[index]:
                    changed_left_boundary = True
                    fill_coords(
                        img[ymin:ymax, xmin:xmax, :],
                        point_in_rect(0, 0.093, 0, 1),
                        color,
                    )
    # use default boundary color if cell is not on boundary of selfish region
    if not changed_left_boundary:
        fill_coords(
            img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 0.031, 0, 1), (100, 100, 100)
        )
    if not changed_top_boundary:
        fill_coords(
            img[ymin:ymax, xmin:xmax, :], point_in_rect(0, 1, 0, 0.031), (100, 100, 100)
        )

    return img


def render_path_line(img, start_pos, end_pos, agent_color, world: WorldT):
    """
    Render a line from current position to target position.
    Uses a simple tile-based approach for efficiency.

    Parameters
    ----------
    img : NDArray
        image of wildfire grid
    start_pos : tuple[int, int]
        starting position (x, y) coordinates
    end_pos : tuple[int, int]
        ending position (x, y) coordinates
    agent_color : str
        color of the agent (used for the line)
    world : WorldT
        wildfire world

    Returns
    -------
    img : NDArray
        image with path line rendered
    """
    tile_size = TILE_PIXELS
    x0, y0 = start_pos
    x1, y1 = end_pos

    color = world.COLORS[agent_color]

    # Bresenham's line algorithm (grid-based)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    x, y = x0, y0

    while True:
        # Draw a small indicator in this tile
        # Skip start and end tiles (already have agent/marker)
        if not ((x == x0 and y == y0) or (x == x1 and y == y1)):
            ymin = y * tile_size + int(tile_size * 0.4)
            ymax = y * tile_size + int(tile_size * 0.6)
            xmin = x * tile_size + int(tile_size * 0.4)
            xmax = x * tile_size + int(tile_size * 0.6)

            # Draw small dot
            alpha = 0.4
            img[ymin:ymax, xmin:xmax] = (
                alpha * color + (1 - alpha) * img[ymin:ymax, xmin:xmax]
            ).astype(np.uint8)

        if x == x1 and y == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return img


def render_target_marker(img, target_pos, agent_color, world: WorldT):
    """
    Render a marker at the target position to show where the agent is heading.

    Parameters
    ----------
    img : NDArray
        image of wildfire grid
    target_pos : tuple[int, int]
        target position (x, y) coordinates
    agent_color : str
        color of the agent (used for the marker)
    world : WorldT
        wildfire world

    Returns
    -------
    img : NDArray
        image with target marker rendered
    """
    tile_size = TILE_PIXELS
    x, y = target_pos
    ymin = y * tile_size
    ymax = (y + 1) * tile_size
    xmin = x * tile_size
    xmax = (x + 1) * tile_size

    color = world.COLORS[agent_color]

    # Draw a small circle at center
    fill_coords(
        img[ymin:ymax, xmin:xmax, :],
        point_in_circle(0.5, 0.5, 0.15),
        color,
    )

    # Draw crosshair
    # Horizontal line
    fill_coords(
        img[ymin:ymax, xmin:xmax, :],
        point_in_rect(0.3, 0.7, 0.48, 0.52),
        color,
    )
    # Vertical line
    fill_coords(
        img[ymin:ymax, xmin:xmax, :],
        point_in_rect(0.48, 0.52, 0.3, 0.7),
        color,
    )

    return img


def render_debug_panel(img, agents, step_count=0):
    """
    Render debug information panel on the right side of the image (Phase 5).
    Extends the image to add panel separately without covering the map.

    Parameters
    ----------
    img : NDArray
        image of wildfire grid
    agents : list[Agent]
        list of agents to display info for
    step_count : int
        current step count

    Returns
    -------
    img : NDArray
        extended image with debug panel on the right
    """
    # Panel dimensions
    panel_width = 200
    panel_padding = 10

    # Create expanded canvas
    old_height, old_width = img.shape[:2]
    new_width = old_width + panel_width
    expanded_img = np.zeros((old_height, new_width, 3), dtype=np.uint8)

    # Copy original image to left side
    expanded_img[:, :old_width, :] = img

    # Fill panel area with dark background
    expanded_img[:, old_width:, :] = [30, 30, 30]

    # Convert to PIL for text rendering
    pil_img = Image.fromarray(expanded_img)
    draw = ImageDraw.Draw(pil_img)

    # Try to use a font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except:
        font = ImageFont.load_default()
        font_bold = font

    # Panel position (in the extended area)
    panel_x = old_width + panel_padding
    panel_y = 10
    line_height = 15

    # Draw step count
    draw.text((panel_x, panel_y), f"Step: {step_count}", fill=(255, 255, 255), font=font_bold)
    panel_y += line_height * 2

    # Draw info for each agent
    for agent in agents:
        # State names
        state_names = {0: "ACTIVE", 1: "RETURNING", 2: "RECHARGING"}
        state = state_names.get(agent.state, "UNKNOWN")

        # Color indicator
        color_map = {"red": (255, 0, 0), "blue": (0, 0, 255), "light_blue": (0, 200, 255)}
        agent_color = color_map.get(agent.color, (255, 255, 255))

        # Agent header
        draw.text((panel_x, panel_y), f"Agent {agent.index}:", fill=agent_color, font=font_bold)
        panel_y += line_height

        # Position
        draw.text((panel_x + 5, panel_y), f"Pos: {agent.pos}", fill=(200, 200, 200), font=font)
        panel_y += line_height

        # Target
        if agent.target_pos:
            dist = np.linalg.norm(np.array(agent.pos) - np.array(agent.target_pos))
            draw.text((panel_x + 5, panel_y), f"Target: {agent.target_pos}", fill=(200, 200, 200), font=font)
            panel_y += line_height
            draw.text((panel_x + 5, panel_y), f"Dist: {dist:.1f}", fill=(200, 200, 200), font=font)
            panel_y += line_height

        # State
        state_color = (0, 255, 0) if agent.state == 0 else (255, 255, 0) if agent.state == 1 else (255, 0, 0)
        draw.text((panel_x + 5, panel_y), f"State: {state}", fill=state_color, font=font)
        panel_y += line_height

        # Active time
        time_percent = int(100 * agent.active_time_remaining / agent.max_active_time)
        draw.text((panel_x + 5, panel_y), f"Active: {agent.active_time_remaining}/{agent.max_active_time} ({time_percent}%)",
                  fill=(200, 200, 200), font=font)
        panel_y += line_height

        # Recharge time (if applicable)
        if agent.state == 2:  # RECHARGING
            recharge_percent = int(100 * (agent.recharge_time - agent.recharge_time_remaining) / agent.recharge_time)
            draw.text((panel_x + 5, panel_y), f"Recharge: {agent.recharge_time_remaining}/{agent.recharge_time} ({recharge_percent}%)",
                      fill=(255, 165, 0), font=font)
            panel_y += line_height

        # Water remaining
        if hasattr(agent, 'max_water') and hasattr(agent, 'water_remaining'):
            water_percent = int(100 * agent.water_remaining / agent.max_water)
            draw.text((panel_x + 5, panel_y), f"Water: {agent.water_remaining:.1f}/{agent.max_water:.1f} ({water_percent}%)",
                      fill=(0, 255, 255), font=font)
            panel_y += line_height

        panel_y += line_height * 0.5  # Space between agents

    # Convert back to numpy array
    return np.array(pil_img)


def get_initial_fire_coordinates(x, y, grid_size, fire_size):
    """Generate the coordinates of trees on fire in a uniformly randomly located square fire region of specified size.

    Parameters:
    -------
        x : int
            the x-coordinate of the center cell of the fire region if n is odd, or the x-coordinate of the top-left corner cell of the fire region if n is even.
        y : int
            the y-coordinate of the center cell of the fire region if n is odd, or the y-coordinate of the top-left corner cell of the fire region if n is even.
        grid_size : int
            the size of square gridworld. Note that the gridworld includes walls.
        fire_size : int
            the side of the square fire region.
    Returns:
    -------
    coordinates : list(tuple(int, int))
        a list of tuples, where each tuple represents the position coordinates of a tree on fire in the fire region.
    """

    if fire_size % 2 == 0:
        # side of the fire region is an even number
        coordinates = []
        # loop through the positions of cells in the fire region. The top-left corner cell is (x, y)
        for i in range(x, x + fire_size):
            for j in range(y, y + fire_size):
                coordinates.append((i, j))
        return coordinates
    else:
        # side of the fire region is an odd number
        # offset is the distance from the center cell to the edge cell of a fire region of size n by n
        offset = int((fire_size - 1) / 2)

        # determine range of x and y coordinates lying within the fire region. The center cell is (x, y). The subtraction of 2 from grid size is due to the walls.
        start_x = max(1, x - offset)
        end_x = min(grid_size - 2, x + offset)
        start_y = max(1, y - offset)
        end_y = min(grid_size - 2, y + offset)

        coordinates = []
        # loop through the positions of cells in the fire region
        for x in range(start_x, end_x + 1):
            for y in range(start_y, end_y + 1):
                coordinates.append((x, y))
        return coordinates
