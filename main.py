from maze_env import Maze
from RL_brain import QLearningTable
import time
def update():
    for episode in range(100):
        # 初始化 state 的观测值
        star=time.perf_counter()
        observation = env.reset()    # 当前位置坐标
        step_count = 0
        while True:
            # 更新可视化环境

            env.render()   # 刷新环境，为了与环境互动

            # RL 大脑根据 state 的观测值挑选 action
            action = RL.choose_action(str(observation))

            # 探索者在环境中实施这个 action, 并得到环境返回的下一个 state 观测值, reward 和 done (是否是掉下地狱或者升上天堂)
            observation_, reward, done , is_sucess = env.step(action)   # 动作执行后更新坐标，奖励已经判断是否触发陷阱
            step_count = step_count + 1
            # RL 从这个序列 (state, action, reward, state_) 中学习
            RL.learn(str(observation), action, reward, str(observation_))

            # 将下一个 state 的值传到下一次循环
            observation = observation_

            # 如果掉下地狱或者升上天堂, 这回合就结束了
            if done:
                end = time.perf_counter()
                qtable_out = RL.redata_qtable()
                print('episode: {}, step count: {},time:{},result:{}\n q-table:\n{}\n'.format(episode, step_count,round(end-star), is_sucess,
                qtable_out))
                break

    # 结束游戏并关闭窗口
    print('game over')
    env.destroy()

if __name__ == "__main__":
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    env.after(100, update)
    env.mainloop()