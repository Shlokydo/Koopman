import copy
import random
import numpy as np 
import pandas as pd 
import random as r
import os
import sys
import Helperfunction as helpfunc

#For plotting
from matplotlib import pyplot as plt
from matplotlib.colors import cnames
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from mpl_toolkits.axisartist.axislines import SubplotZero

def plot_figure(x_t, save_image):
    fig = plt.figure()
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
#     ax = fig.add_axes([0, 0, 1, 1], projection = None)
#     ax.axis('on')
    
    for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)
    
    for direction in ["left", "right", "bottom", "top"]:
    # hides borders
        ax.axis[direction].set_visible(False)
    
    # Turn off tick    
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    # prepare the axes limits
    ax.set_xlim((np.min(x_t[:,:,0])-0.5, np.max(x_t[:,:,0])+0.5))
    ax.set_ylim((np.min(x_t[:,:,1])-0.5, np.max(x_t[:,:,1])+0.5))

    # choose a different color for each trajectory
    color_count = np.linspace(0,1,x_t.shape[0]/2)
    concat_list = [color_count]
    num_traj = int(x_t.shape[0]/2)
    concat_list = 2 * concat_list
    concat_list = np.concatenate((concat_list))
    colors = plt.cm.viridis(concat_list)
    # set up lines and points
    lines = sum([ax.plot([], [], '-', c=c)
                for c in colors], [])
    pts = sum([ax.plot([], [], 'o', c=c)
            for c in colors], [])

    for j in range(int(x_t.shape[0]/2)):
        lines = ax.plot(x_t[j,:,0], x_t[j,:,1], '-', label = f'Trajectory {j+1}', c=colors[j])
        plt.setp(lines, linewidth=1)
        lines = ax.plot(x_t[j+int(x_t.shape[0]/2),:,0], x_t[j+int(x_t.shape[0]/2),:,1], '-*', c=colors[j])
        plt.setp(lines, linewidth=1)

    ax.margins(0.001)
    ax.legend()
    plt.show()
    if save_image == True:
        fig.savefig('./nl_pendulum.png', format= 'png', dpi = 1200)
    return

def plot_diff(x_t, time, save_image, name):
    fig = plt.figure()
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    
    for direction in ["xzero", "yzero"]:
    # adds arrows at the ends of each axis
        ax.axis[direction].set_axisline_style("-|>")

    # adds X and Y-axis from the origin
        ax.axis[direction].set_visible(True)
    
    for direction in ["left", "right", "bottom", "top"]:
    # hides borders
        ax.axis[direction].set_visible(False)

    # prepare the axes limits
    ax.set_xlim((np.min(time), np.max(time)+0.5))
    ax.set_ylim((np.min(x_t)-0.05, np.max(x_t)+0.05))

    # choose a different color for each trajectory
    colors = plt.cm.viridis(np.linspace(0, 1, x_t.shape[0]))

    for i in range(1):
        x = x_t
        for j in range(x.shape[0]):
            lines = ax.plot(time[:x.shape[1]], x[j,:], '-*', c=colors[j], label=f'Trajectory {j+1}')
            plt.setp(lines, linewidth=1)
            

    ax.margins(0.001)
    ax.legend()
    plt.grid(True)
    plt.show()
    if save_image == True:
        fig.savefig(name, format= 'png', dpi = 1200)
    return

def animate(x_t):
    fig = plt.figure()
    ax = fig.add_axes([0, 0, 1, 1], projection= None)
    ax.axis('off')

    # choose a different color for each trajectory
    color_count = np.linspace(0,1,x_t.shape[0]/2)
    concat_list = [color_count]
    num_traj = int(x_t.shape[0]/2)
    concat_list = 2 * concat_list
    concat_list = np.concatenate((concat_list))
    colors = plt.cm.viridis(concat_list)
    # set up lines and points
    lines = sum([ax.plot([], [], '-', c=c)
                for c in colors], [])
    pts = sum([ax.plot([], [], 'o', c=c)
            for c in colors], [])

    # prepare the axes limits
    ax.set_xlim((np.min(x_t[:,:,0])-0.5, np.max(x_t[:,:,0])+0.5))
    ax.set_ylim((np.min(x_t[:,:,1])-0.5, np.max(x_t[:,:,1])+0.5))

    # initialization function: plot the background of each frame
    def init():
        for line, pt in zip(lines, pts):
            line.set_data([], [])
            pt.set_data([], [])
        return lines + pts

    # animation function.  This will be called sequentially with the frame number
    def animate(i):
        # we'll step two time-steps per frame.  This leads to nice results.
        i = (1 * i) % x_t.shape[1]

        for line, pt, xi in zip(lines, pts, x_t):
            x, y = xi[:i].T
            line.set_data(x, y)
            pt.set_data(x[-1:], y[-1:])
        fig.canvas.draw()
        return lines + pts

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                frames=70, interval=30, blit=True)

    # Save as mp4. This requires mplayer or ffmpeg to be installed
    anim.save('./NL_Pendulum_video.mp4', fps=15, extra_args=['-vcodec', 'libx264'], dpi = 235)