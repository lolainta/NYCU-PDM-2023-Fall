import numpy as np
from icecream import ic
from matplotlib import pyplot as plt
from enums import Objects
from pick import pick
from tqdm import tqdm


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
    color01: np.array, color0255: np.array, point: np.array, constraint: tuple
):
    ret01, ret0255, retpoint = [], [], []
    for c01, c0255, p in zip(color01, color0255, point):
        if np.all(c0255 == constraint):
            continue
        ret01.append(c01)
        ret0255.append(c0255)
        retpoint.append(p)
    return np.array(ret01), np.array(ret0255), np.array(retpoint)


# def part1():
def main():
    col01, col255, point = load_data("semantic_3d_pointcloud")
    col01, col255, point = filter_data(col01, col255, point, Objects.ceiling.value)
    col01, col255, point = filter_data(col01, col255, point, Objects.floor.value)

    # ic(col01.shape, col255.shape, point.shape)
    fig, ax = plt.subplots()
    fig.set_figheight(10)
    fig.set_figwidth(10)
    ax.scatter(point[:, 0], point[:, 2], c=col01)
    # ax.axis("off")

    def onclick(event):
        start = event.xdata, 0, event.ydata
        print(
            f"button={event.button}, x={event.x}, y={event.y}, xdata={event.xdata}, ydata={event.ydata}"
        )
        fig.canvas.mpl_disconnect(cid)
        ax.scatter(start[0], start[2], c="green")
        ax.scatter(end[0], end[2], c="blue")
        print(start, end)
        print("Press q to quit.")
        plt.axis("scaled")
        plt.show()
        plt.cla()
        rrt_raw(start, end, point, col255)

    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    start, end = None, (1, 2, 3)
    # show

    title = "Please choose an target object: "
    # options = [obj.name for obj in Objects]
    options = ["rack", "cushion", "lamp", "stair", "cooktop"]
    option, _ = pick(options, title)
    color = Objects[option].value
    print(f"Selected {option} as target object. Color: {color}")
    end = None
    for c, p in zip(col255, point):
        if np.all(c == color):
            end = p
            break
    print("Please click on the figure to select the start point.")
    plt.axis("scaled")
    plt.show()
    plt.cla()


def part2(start, end):
    pass


def rrt_raw(src, dst, point, color):
    xmin = np.min(point[:, 0])
    xmax = np.max(point[:, 0])
    ymin = np.min(point[:, 2])
    ymax = np.max(point[:, 2])
    grid_size = 0.02
    grid = np.zeros(
        (int((xmax - xmin) / grid_size) + 1, int((ymax - ymin) / grid_size) + 1)
    )
    print(grid.shape)

    def point_to_grid(p):
        return int((p[0] - xmin) / grid_size), int((p[2] - ymin) / grid_size)

    for p in point:
        grid[point_to_grid(p)] = 1

    start = point_to_grid(src)
    end = point_to_grid(dst)
    grid[start] = 2
    grid[end] = 3
    print(start, end)
    tqdm.write("Drawing grid map")
    for p in tqdm(np.argwhere(grid == 1)):
        if np.random.rand() > 0.1:
            continue
        plt.scatter(p[0], p[1], c="black")
    plt.scatter(start[0], start[1], c="green")
    plt.scatter(end[0], end[1], c="blue")
    plt.show()
    rrt_grid(grid, start, end)


def rrt_grid(grid, start, end):
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

    def steer(nearest, x, y):
        dx = x - nearest[0]
        dy = y - nearest[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 15:
            dx = dx / dist * 9
            dy = dy / dist * 9
        return int(nearest[0] + dx), int(nearest[1] + dy)

    vertices = [start]
    edges = []
    parent = {}
    pbar = tqdm(total=1000, desc="RRT")
    while True:
        # sample
        x, y = random_sample()
        if grid[x, y] == 1:
            continue

        nearest = find_nearest(x, y)
        if nearest is None:
            continue
        # steer
        new = steer(nearest, x, y)

        # check collision
        if grid[new] == 1:
            continue
        # add
        vertices.append(new)
        edges.append((nearest, new))
        parent[new] = nearest
        pbar.update(1)
        # check end
        # if pbar.n >= pbar.total:
        #     tqdm.write("Failed to find path!")
        #     break
        if distance(new, end) < 15:
            parent[end] = new
            tqdm.write("Found path!")
            break
    pbar.close()

    path = []
    cur = end
    while True:
        path.append(cur)
        if cur == start:
            break
        cur = parent[cur]
    print(path)
    tqdm.write("Drawing RRT")
    for p in tqdm(path):
        plt.scatter(p[0], p[1], c="yellow")
    for e in tqdm(edges):
        plt.plot([e[0][0], e[1][0]], [e[0][1], e[1][1]], c="red")
    plt.axis("scaled")
    plt.show()


def test():
    fig, ax = plt.subplots()
    ax.plot(np.random.rand(10))
    plt.show()


if __name__ == "__main__":
    np.random.seed(0)
    main()
