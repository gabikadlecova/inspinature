import numpy as np
import matplotlib.pyplot as plt
from IPython import display


def init_render(env):
    return plt.imshow(env.render())


def plot_step(env, img):
    # plotting stuff (jupyter specific)
    render_data = env.render()
    img.set_data(render_data) # just update the data
    display.display(plt.gcf())
    display.clear_output(wait=True)


def show_animation(agent, env, steps=200, episodes=1, is_jupyter=True):
    ''' Pomocna funkce, ktera zobrazuje chovani zvoleneho agenta v danem 
    prostredi.
    Parameters
    ----------
    agent: 
        Agent, ktery se ma vizualizivat, musi implementovat metodu
        act(observation, reward, done)
    env:
        OpenAI gym prostredi, ktere se ma pouzit
    
    steps: int
        Pocet kroku v prostredi, ktere se maji simulovat
    
    episodes: int
        Pocet episod, ktere se maji simulovat - kazda a pocet kroku `steps`.
    '''
    if is_jupyter:
        img = init_render(env)
    
    for i in range(episodes):
        obs, info = env.reset()
        done = False
        terminated = False
        R = 0
        t = 0
        r = 0
        while not (done or terminated) and t < steps:
            if is_jupyter:
                plot_step(env, img)
            else:
                env.render()

            action = agent.act(obs, r, done)
            #obs, r, done, _ = env.step(action)
            obs, r, done, terminated, _ = env.step(action)
            R += r
            t += 1
        agent.reset()

def moving_average(x, n):
    weights = np.ones(n)/n
    return np.convolve(np.asarray(x), weights, mode='valid')