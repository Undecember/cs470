import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    timings, ymax = load_timings()
    fig, ax = plt.subplots()
    draw_axis(ax, timings[-1][-1], ymax)
    for timing in timings:
        if timing[2] == 'target' and timing[3] == 'sampling':
            left = None
            for t in timings:
                if t[2] == 'draft' and t[3] == 'sampling' and t[0] == timing[0]:
                    left = t[4]
                    break
            if left is not None:
                draw_rect(ax, ((left, timing[0] - 1), (timing[5], timing[1] - 1)),
                          "#ff0000")
    for timing in timings:
        draw_rect(ax, ((timing[4], timing[0] - 1), (timing[5], timing[1] - 1)),
                  "#fca503" if timing[2] == 'draft' else "#76d925")
    fig.show()
    input()

def load_timings():
    res = []
    with open(sys.argv[1]) as f:
        while True:
            line = f.readline()
            if line == '': break
            res += [ line.split() ]
    ymax = -1
    for i in range(len(res)):
        res[i][0] = int(res[i][0])
        res[i][1] = int(res[i][1])
        ymax = ymax if ymax > res[i][1] else res[i][1]
        res[i][4] = int(res[i][4]) / 1000
        res[i][5] = int(res[i][5]) / 1000
    return res, ymax - 1

def draw_axis(ax, x_max, y_max):
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

def draw_rect(ax, coords, color):
    (x1, y1), (x2, y2) = coords
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle(
        (x1, y1), width, height, linewidth=1, edgecolor='none', facecolor=color)
    ax.add_patch(rect)

def save_img(fig, filename):
    fig.savefig(filename)

if __name__ == "__main__":
    main()
