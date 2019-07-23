import numpy as np
import time
import sys
import math
from random import uniform

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import itertools

UNIT = 100  # pixels
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        # a = [[.1, .4, .6, -.1, -.4, -.6]]
        # b = [[.1, .4, .6, -.1, -.4, -.8]]
        a = [[.2, .4, .6, -.2, -.4, -.6]]
        b = [[.2, .4, .6, -.2, -.4, -.6]]

        c = np.vstack((a, b))
        self.data = list(itertools.product(*c))
        self.data = [str(x) for x in self.data]

        self.action_space = self.data

        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))

        self._vspeed = self.my_round(uniform(-1, 1), 50)
        self._xdistance = self.my_round(uniform(-0.6, 0.6), 100)

        self.m = 0.5
        self.g = 9.8
        self.l = 2
        self.c = 0.01
        self.t = 0.02
        self.table_width = MAZE_W * UNIT * .8 - MAZE_W * UNIT * .2
        self.table_height_move = MAZE_W * UNIT * .8 - MAZE_W * UNIT * .2
        self._build_maze()

        self._time = 0

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        origin = np.array([MAZE_W * UNIT * .5, MAZE_H * UNIT * .5 - 20])

        oval_center = origin + self._xdistance * (self.table_width / 2)

        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, origin[1] - 15,
            oval_center[0] + 15, origin[1] + 15,
            fill='red')

        self.flatTable = self.canvas.create_line(MAZE_W * UNIT * .2, MAZE_H * UNIT * .5, MAZE_W * UNIT * .8,
                                                 MAZE_H * UNIT * .5, fill='#531B0A',
                                                 width=10)
        self.zero = self.canvas.create_line(MAZE_W * UNIT * .5, MAZE_H * UNIT * .5 - 10, MAZE_W * UNIT * .5,
                                            MAZE_H * UNIT * .5 + 10, fill='red',
                                            width=2)
        # pack all

        # self.gif1 = tk.PhotoImage(file='./Pic/robot1.gif', format="gif -index 2")
        self._numr1 = 0
        self._numr2 = 0
        self._animateagent2()
        self._animateagent1()

        bg = tk.PhotoImage(file='./Pic/bg.png')
        self.bg = self.canvas.create_image(0, 0, image=bg, anchor=tk.NW)
        self.robot1 = self.canvas.create_image(30, MAZE_H * UNIT * .5 - 50, image=self.gif1, anchor=tk.NW)
        self.robot2 = self.canvas.create_image(MAZE_W * UNIT - 80, MAZE_H * UNIT * .5 - 50, image=self.gif2,
                                               anchor=tk.NW)

        # label = tk.Label(image=gif1)
        # label.configure(image=gif1)
        # label.image = gif1  # keep a reference!
        label = tk.Label(image=bg)
        label.image = bg

        # image = ImageTk.PhotoImage(file="./Pic/bg.png")
        # self.canvas.create_image(0, 0, image=image, anchor=tk.NW)
        # self.canvas.pack(expand=tk.YES, fill=tk.BOTH)

        self.canvas.pack()

    def _animateagent2(self):
        try:
            self.gif2 = tk.PhotoImage(file="./Pic/robot2.gif",
                                      format='gif -index {}'.format(self._numr2))  # Looping through the frames
            label = tk.Label(image=self.gif2)
            label.configure(image=self.gif2)
            self.robot2 = self.canvas.create_image(MAZE_W * UNIT - 80, MAZE_H * UNIT * .5 - 50, image=self.gif2,
                                                   anchor=tk.NW)

            # label = tk.Label(image=self.gif)
            # label.image = self.gif  # keep a reference!
            self.canvas.pack()
            self._numr2 += 1
        except tk.TclError:  # When we try a frame that doesn't exist, we know we have to start over from zero
            self._numr2 = 0

        self.after(int(.1 * 1000), self._animateagent2)

    def _animateagent1(self):
        try:
            self.gif1 = tk.PhotoImage(file="./Pic/robot1.gif",
                                      format='gif -index {}'.format(self._numr1))  # Looping through the frames
            label = tk.Label(image=self.gif1)
            label.configure(image=self.gif1)
            self.robot1 = self.canvas.create_image(30, MAZE_H * UNIT * .5 - 50, image=self.gif1, anchor=tk.NW)

            # label = tk.Label(image=self.gif)
            # label.image = self.gif  # keep a reference!
            self.canvas.pack()
            self._numr1 += 1
        except tk.TclError:  # When we try a frame that doesn't exist, we know we have to start over from zero
            self._numr1 = 0

        self.after(int(.1 * 1000), self._animateagent1)

    def reset(self):
        self.update()
        self._time = 0
        time.sleep(0.7)
        self._vspeed = self.my_round(uniform(-1, 1), 50)
        self._xdistance = self.my_round(uniform(-0.6, 0.6), 100)
        self.canvas.delete(self.oval)
        self.canvas.delete(self.flatTable)
        origin = np.array([MAZE_W * UNIT * .5, MAZE_H * UNIT * .5 - 20])
        # create oval

        oval_center = origin + self._xdistance * (self.table_width / 2)

        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, origin[1] - 15,
            oval_center[0] + 15, origin[1] + 15,
            fill='red')
        self.flatTable = self.canvas.create_line(MAZE_W * UNIT * .2, MAZE_H * UNIT * .5, MAZE_W * UNIT * .8,
                                                 MAZE_H * UNIT * .5, fill='black',
                                                 width=8)

        return self._xdistance, self._vspeed

    def nextState(self, v, h1, h2):
        xresult = ((-self.c * v) + (self.m * self.g * ((h1 - h2) / self.l))) / self.m
        return xresult

    def reward(self, x, xbar):
        p = 0.25 * 0.25
        return 0.8 * np.exp(-(x * x) / p) + 0.2 * np.exp(-(xbar * xbar) / p)

    def reward2(self, x, xbar):
        return 1 - x * x - xbar * xbar

    def xPos(self, a, t, v, x_0):
        return 1 / 2 * a * t * t + v * t + x_0

    def vSpeed(self, a, t, v_0):
        return a * t + v_0

    def my_round(self, x, p):
        return round(x * p) / p

    def MN(self, AN, BC, AB):
        return AN * BC / math.sqrt(math.fabs(AB * AB - BC * BC))

    def movextable(self, h1, h2):
        h = h1 - h2
        return math.sqrt(4 - h * h)

    def step(self, state, action):

        s = self.canvas.coords(self.oval)
        base_action = np.array([0.0, 0.0])
        X = state

        a = self.nextState(X[1], float(action[0]), float(action[1]))
        base_action[0] = self.xPos(a, self.t, X[1], X[0])
        v = self.vSpeed(a, self.t, X[1])

        tw = (self.table_width / 2)

        # move Ball
        moveTableHeight = self.movextable(action[0], action[1]) / 2
        AN = self.canvas.coords(self.oval)[0] - self.canvas.coords(self.flatTable)[0]
        AC = moveTableHeight * 210
        BC = (action[0] - action[1]) * (self.table_height_move / 2)
        base_action[1] = (AN * BC / AC) - action[0] * (self.table_height_move / 2)
        base_action[1] -= (s[3] - MAZE_H * UNIT * .5)
        movex = ((base_action[0] * tw) - (s[0] - MAZE_W * UNIT * .5))
        moveh = base_action[1]
        self.canvas.move(self.oval, movex, moveh)  # move agent

        # round data to discrete
        base_action[0] = self.my_round(base_action[0], 100)
        v = self.my_round(v, 50)
        # print(v)
        if v < -1:
            v = -1
        if v > 1:
            v = 1

        # Update table
        self.canvas.delete(self.flatTable)
        self.flatTable = self.canvas.create_line(
            MAZE_W * UNIT * .2 + (self.table_width / 2) - (moveTableHeight * (self.table_width / 2)),
            (-action[0] * (self.table_height_move / 2) + UNIT * MAZE_H * .5),
            (MAZE_W * UNIT * .8) - ((self.table_width / 2) - (moveTableHeight * (self.table_width / 2))),
            (-action[1] * (self.table_height_move / 2) + UNIT * MAZE_H * .5),
            fill='#531B0A',
            width=10)

        s = self.canvas.coords(self.oval)
        ss = self.canvas.coords(self.flatTable)

        s_ = (base_action[0], v)

        # print(X, s_ , action, a)
        reward = self.reward2(base_action[0], v)

        # check ball into table
        if s[2] > ss[2] + 15 or s[0] < ss[0] - 15:
            # print(X, action, a)
            done = True
            self.canvas.move(self.oval, 0, 50)  # move agent
            self._time = 0
        else:
            done = False
        return s_, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()


def update():
    for t in range(10):
        env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
