import numpy as np
from icecream import ic
from matplotlib import pyplot as plt
from enums import Objects
from pick import pick
from tqdm import tqdm
from nav import Navigator


def load_data(folder: str):
    with open(f"{folder}/color01.npy", "rb") as f:
        color01 = np.load(f)
    with open(f"{folder}/color0255.npy", "rb") as f:
        color0255 = np.load(f)
    with open(f"{folder}/point.npy", "rb") as f:
        point = np.load(f)
    point = point * 10000 / 255
    return color01, color0255, point


def filter_data(
    color01: np.array, color0255: np.array, point: np.array, constraint: Objects
):
    ret01, ret0255, retpoint = [], [], []
    for c01, c0255, p in zip(color01, color0255, point):
        if np.all(c0255 == constraint.value):
            continue
        ret01.append(c01)
        ret0255.append(c0255)
        retpoint.append(p)
    return np.array(ret01), np.array(ret0255), np.array(retpoint)


def filter_non_ground(color01: np.array, color0255: np.array, point: np.array):
    ret01, ret0255, retpoint = [], [], []
    for c01, c0255, p in zip(color01, color0255, point):
        if p[1] > 0.5:
            continue
        ret01.append(c01)
        ret0255.append(c0255)
        retpoint.append(p)
    return np.array(ret01), np.array(ret0255), np.array(retpoint)


def main():
    col01, col255, point = load_data("semantic_3d_pointcloud")
    col01, col255, point = filter_data(col01, col255, point, Objects.ceiling)
    col01, col255, point = filter_data(col01, col255, point, Objects.floor)
    col01, col255, point = filter_non_ground(col01, col255, point)
    scale = 20
    for x in range(int(2.2 * scale), int(4 * scale)):
        for y in range(int(4.8 * scale), int(5.8 * scale)):
            col01 = np.append(col01, [[1, 1, 1]], axis=0)
            col255 = np.append(col255, [[255, 255, 255]], axis=0)
            point = np.append(point, [[x / scale, 0, y / scale]], axis=0)

    ic(col01.shape, col255.shape, point.shape)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax.scatter(point[:, 0], point[:, 2], c=col01, s=0.1)
    ax.axis("off")
    ax.axis("scaled")
    fig.savefig("map.png")

    def onclick(event):
        start = event.xdata, 0, event.ydata
        ic(
            f"button={event.button}, x={event.x}, y={event.y}, xdata={event.xdata}, ydata={event.ydata}"
        )
        fig.canvas.mpl_disconnect(cid)
        ax.cla()
        ax.scatter(point[:, 0], point[:, 2], c=col01, s=0.1)
        ax.scatter(start[0], start[2], c="green", s=20)
        ax.scatter(end[0], end[2], c="blue", s=20)
        fig.show()
        ic(start, end)
        print("Press q to quit.")
        rrt_raw(start, end, point, col01, tar, fig, ax)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    start, end = None, None

    title = "Please choose an target object: "
    # options = [obj.name for obj in Objects]
    options = ["rack", "cushion", "lamp", "stair", "cooktop"]
    option, _ = pick(options, title)
    tar = Objects[option]
    print(f"Selected {option} as target object. Color: {tar.value}")
    end = None
    if option == "rack":
        end = 3.5, 0, 3.8
    elif option == "cushion":
        end = 1.3, 0, 8.5
    elif option == "lamp":
        end = 0, 0, 4.3
    elif option == "stair":
        end = 5, 0, 7.5
    elif option == "cooktop":
        end = -0.8, 0, -1.5
    else:
        raise Exception("Invalid target object")
    print("Please click on the figure to select the start point.")
    fig.show()
    fig.waitforbuttonpress()


def rrt_raw(src, dst, point, col01, tar: Objects, fig, ax):
    xmin = np.min(point[:, 0])
    xmax = np.max(point[:, 0])
    ymin = np.min(point[:, 2])
    ymax = np.max(point[:, 2])
    grid_size = 0.01
    thick = 12
    step = 0.3 / grid_size
    grid = np.zeros(
        (
            int((xmax - xmin) / grid_size) + thick + 1,
            int((ymax - ymin) / grid_size) + thick + 1,
        )
    )
    ic(grid.shape)

    def point_to_grid(p):
        return int((p[0] - xmin) / grid_size), int((p[2] - ymin) / grid_size)

    for p in point:
        gp = point_to_grid(p)
        for i in range(thick):
            grid[gp[0] + i, gp[1]] = 1
            grid[gp[0] - i, gp[1]] = 1
            grid[gp[0], gp[1] + i] = 1
            grid[gp[0], gp[1] - i] = 1
            grid[gp[0] + i, gp[1] + i] = 1
            grid[gp[0] - i, gp[1] - i] = 1
            grid[gp[0] + i, gp[1] - i] = 1
            grid[gp[0] - i, gp[1] + i] = 1

    start = point_to_grid(src)
    end = point_to_grid(dst)
    grid[start] = 2
    grid[end] = 3
    ic(start, end)
    vertices, edges, path = rrt_grid(grid, start, end, step)

    ic(len(vertices), len(edges), len(path))

    def grid_to_point(p):
        return p[0] * grid_size + xmin, 0, p[1] * grid_size + ymin

    vertices = np.array([grid_to_point(p) for p in vertices])
    path = np.array([grid_to_point(p) for p in path])
    edges = np.array([(grid_to_point(p1), grid_to_point(p2)) for p1, p2 in edges])
    ic(path.shape, vertices.shape, edges.shape)
    ax.scatter(vertices[:, 0], vertices[:, 2], c="black", s=0.1)
    for e in edges:
        ax.plot([e[0][0], e[1][0]], [e[0][2], e[1][2]], c="black", linewidth=0.1)
    for i in tqdm(range(len(path) - 1)):
        ax.plot(
            [path[i, 0], path[i + 1, 0]],
            [path[i, 2], path[i + 1, 2]],
            c="red",
            linewidth=3,
        )
    # fig.show()
    fig.savefig(f"results/{tar.name}.png")
    # fig.waitforbuttonpress()

    visualize(path, tar)


def rrt_grid(grid: np.array, start, end, step):
    def distance(a, b):
        return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def random_sample():
        return np.random.randint(grid.shape[0]), np.random.randint(grid.shape[1])

    def find_nearest(x, y):
        nearest = None
        min_dist = 1e9
        for v in vertices:
            dist = np.sqrt((v[0] - x) ** 2 + (v[1] - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = v
        return nearest

    def steer(nearest, x, y, step):
        dx = x - nearest[0]
        dy = y - nearest[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > step:
            dx = dx / dist * step
            dy = dy / dist * step
        return int(nearest[0] + dx), int(nearest[1] + dy)

    vertices = [start]
    edges = []
    parent = {start: start}
    ic(grid.size)
    pbar = tqdm(total=grid.size, desc="RRT")
    while True:
        # sample
        x, y = random_sample()
        if grid[x, y] == 1:
            continue

        nearest = find_nearest(x, y)
        if nearest is None:
            continue
        # steer
        new = steer(nearest, x, y, step)

        # check collision
        if grid[new] == 1 or new in vertices:
            continue
        # add
        vertices.append(new)
        edges.append((nearest, new))
        assert new not in parent
        assert nearest in parent
        parent[new] = nearest
        pbar.update(1)
        # check end
        if pbar.n >= pbar.total:
            tqdm.write("Failed to find path!")
            break
        if distance(new, end) < 2 * step:
            parent[end] = new
            tqdm.write("Found path!")
            break
    pbar.close()
    print(len(vertices), len(edges), len(parent))
    path = []
    cur = end
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    path = path[::-1]
    return vertices, edges, path


def visualize(path, tar):
    # ic(path)
    nav = Navigator(path[0], tar)
    # nav.interactive()
    for v in path:
        ic(v)
        nav.goto(v)

        cur = nav.agent.get_state().position
        cur[1] = 0
        error = np.linalg.norm(cur - v)
        ic(error, cur, v)
    input("Press enter to quit.")
    nav.done()

    # nav.interactive()


if __name__ == "__main__":
    ic.disable()
    main()
