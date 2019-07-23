from maze_env import Maze
from RL_brain import QLearningTable
import sys
import pandas as pd


def update():
    try:

        RL.q_table = pd.read_pickle("./Data/dataframe.pk1")
        for episode in range(3000000):
            isRunning = True
            observation = env.reset()
            # if episode % 2000 == 0:
            #     RL.q_table.to_pickle("./Data/dataframe.pk1")
            while isRunning:
                # RL choose action based on observation
                action = RL.choose_action(str(observation))
                # print(observation, len(RL.q_table.index),action)
                for i in range(15):

                    # fresh env
                    env.render()

                    # RL take action and get next observation and reward
                    observation_, reward, done = env.step(observation, eval(action))

                    RL.learn(str(observation), str(action), reward, str(observation_))
                    # swap observation
                    observation = observation_

                    # # break while loop when end of this episode
                    if done:
                        isRunning = False
                        print(episode, len(RL.q_table.index))
                        break
                        # RL learn from this transition



    except KeyboardInterrupt:
        # RL.q_table.to_pickle("./Data/dataframe.pk1")
        sys.exit()
    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=env.action_space)
    env.after(1, update)

env.mainloop()
