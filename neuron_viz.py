import torch
import tkinter as tk
import models
import rospy
import env
import numpy as np


def colorize(num):
    num[num > 1] = 1
    num[num < 0] = 0
    num = (num*255).byte()
    color_codes = []
    for n in num:
        h = format(n, "02x")
        code = "#"+h+h+h
        color_codes.append(code)
    return color_codes


def paint(canvas, coordinates, colors=None):
    for i, (x, y) in enumerate(coordinates):
        canvas.create_rectangle(
            x*SQUARE_SIZE,
            y*SQUARE_SIZE,
            (x+1)*SQUARE_SIZE,
            (y+1)*SQUARE_SIZE,
            fill=colors[i] if colors is not None else "#ffffff")


def find_cool_locations(num_points, start_row, start_col):
    row = start_row
    col = start_col
    locs = []
    for _ in range(num_points):
        locs.append([col, row])
        col += 1
        if col == W_SQUARES:
            col = 0
            row += 1
    return locs


HEIGHT = 720
WIDTH = 1920
SQUARE_SIZE = WIDTH // 64
W_SQUARES = WIDTH // SQUARE_SIZE
H_SQUARES = HEIGHT // SQUARE_SIZE

colors = [
    "deepskyblue",
    "tomato",
    "yellow",
    "maroon1",
    "SeaGreen1",
    "brown4",
    "purple1",
    "cyan",
    "lemonchiffon",
    "gray50"]

object_list = ["white_ball", "red_ball", "yellow_ball"]
rospy.init_node("visualizer")
world = env.Environment(object_list)

master = tk.Tk()

left = tk.Frame(master)
left.pack(side=tk.LEFT)

right = tk.Frame(master)
right.pack(side=tk.RIGHT)

canvas_left = tk.Canvas(left, bg="white", height=HEIGHT, width=WIDTH)
canvas_left.pack()

canvas_right = tk.Canvas(right, bg="white", height=HEIGHT, width=WIDTH//4)
canvas_right.pack()


sd = torch.load("save/policy_net_last.ckpt")
policy_network = models.MLP_gaussian([len(object_list)*2, 32, 32, 32, 2])
policy_network.load_state_dict(sd)
for p in policy_network.parameters():
    p.requires_grad = False


def refresh():
    # x, y = event.x, event.y
    # x = (x * 10) / HEIGHT - 5
    # y = (y * 10) / HEIGHT - 5
    # print(x, ",", y)
    with torch.no_grad():
        inp = torch.tensor(world.get_state(), dtype=torch.float)
        o1 = policy_network.model[:2](inp)
        o2 = policy_network.model[:4](inp)
        o3 = policy_network.model[:6](inp)
        o4 = policy_network(inp)
        o4[..., :o4.shape[-1]//2] = o4[..., :o4.shape[-1]//2] / (2*np.pi) + 0.5

    c1 = colorize(inp)
    c2 = colorize(o1 / o1.max())
    c3 = colorize(o2 / o2.max())
    c4 = colorize(o3 / o3.max())
    c5 = colorize(o4)
    canvas_left.delete("all")
    canvas_right.delete("all")

    RADIUS = 100
    center_x = WIDTH // 8
    center_y = HEIGHT // 2
    canvas_right.create_oval(center_x-RADIUS, center_y-RADIUS, center_x+RADIUS, center_y+RADIUS, fill="#0000ff")
    DOT_RADIUS = 20
    ANGLE = (o4[0].item() * np.pi * 2 - np.pi)
    ANGLE_VAR = o4[1].item()
    # center
    DOT_x = center_x + 200 * np.sin(ANGLE)
    DOT_y = center_y + 200 * np.cos(ANGLE)
    canvas_right.create_oval(DOT_x-DOT_RADIUS, DOT_y-DOT_RADIUS, DOT_x+DOT_RADIUS, DOT_y+DOT_RADIUS, fill="#ff0000")
    # dangle = 2*ANGLE_VAR/50
    # for i in range(50):
    #     DOT_x = center_x + 200 * np.cos(ANGLE+i*dangle-ANGLE_VAR)
    #     DOT_y = center_y + 200 * np.sin(ANGLE+i*dangle-ANGLE_VAR)
    #     canvas_right.create_oval(DOT_x-DOT_RADIUS, DOT_y-DOT_RADIUS, DOT_x+DOT_RADIUS, DOT_y+DOT_RADIUS, fill="#ffcccc", outline="")

    # upper
    DOT_x = center_x + 200 * np.sin(ANGLE+ANGLE_VAR)
    DOT_y = center_y + 200 * np.cos(ANGLE+ANGLE_VAR)
    canvas_right.create_oval(DOT_x-DOT_RADIUS, DOT_y-DOT_RADIUS, DOT_x+DOT_RADIUS, DOT_y+DOT_RADIUS, fill="#ffcccc", outline="")
    # lower
    DOT_x = center_x + 200 * np.sin(ANGLE-ANGLE_VAR)
    DOT_y = center_y + 200 * np.cos(ANGLE-ANGLE_VAR)
    canvas_right.create_oval(DOT_x-DOT_RADIUS, DOT_y-DOT_RADIUS, DOT_x+DOT_RADIUS, DOT_y+DOT_RADIUS, fill="#ffcccc", outline="")

    for y in range(H_SQUARES):
        for x in range(W_SQUARES):
            paint(canvas_left, [[x, y]])

    row = 2
    if len(c1) > W_SQUARES:
        col = 0
    else:
        col = (W_SQUARES-len(c1))//2
    locs = find_cool_locations(len(c1), row, col)
    paint(canvas_left, locs, colors=c1)
    row = locs[-1][1]+2

    if len(c2) > W_SQUARES:
        col = 0
    else:
        col = (W_SQUARES-len(c2))//2
    locs = find_cool_locations(len(c2), row, col)
    paint(canvas_left, locs, colors=c2)
    row = locs[-1][1]+2

    if len(c3) > W_SQUARES:
        col = 0
    else:
        col = (W_SQUARES-len(c3))//2
    locs = find_cool_locations(len(c3), row, col)
    paint(canvas_left, locs, colors=c3)
    row = locs[-1][1]+2

    if len(c4) > W_SQUARES:
        col = 0
    else:
        col = (W_SQUARES-len(c4))//2
    locs = find_cool_locations(len(c4), row, col)
    paint(canvas_left, locs, colors=c4)
    row = locs[-1][1]+2

    if len(c5) > W_SQUARES:
        col = 0
    else:
        col = (W_SQUARES-len(c5))//2
    locs = find_cool_locations(len(c5), row, col)
    paint(canvas_left, locs, colors=c5)
    row = locs[-1][1]+2

    master.after(100, refresh)


master.after(100, refresh)
tk.mainloop()
