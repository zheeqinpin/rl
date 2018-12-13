import gym
from RL_brain import DeepQNetwork
import time

def run_maze(env, RL,n_iter):
    step = 0
    for episode in range(n_iter):
        # initial observation
        _,_,observation = env.reset()
#        print(type(observation))
        print('No. %d,the initial state is %s' % (episode,observation))
        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            n_action = RL.choose_action(observation)
            action = env.acdict[n_action]
            # RL take action and get next observation and reward
            next_state, reward, done, info = env.step(action)
            observation_ = env.scdict[next_state]
            RL.store_transition(observation, n_action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
#            print('N0. %d' % step)
            step += 1

    # end of game
    print('End of Training!!!')
def run_test(env, RL):
    state = env.myreset()
    RL.reloadPara()
    while True:
        env.render()       #显示创建的环境  
        print('current state is %s' % state)
        n_action = RL.choose_best_action(state)
        action = env.acdict[n_action]
        print(action)
        # RL take action and get next observation and reward
        next_state, reward, done, info = env.step(action)
        print('next-state is %s' % next_state)
        state = env.scdict[next_state]
        time.sleep(0.5)
        if done:
            print('Game Over!!!')
            env.render()
            break
def main():
    env = gym.make('MazeEnv-v0')
    env = env.unwrapped #消除限制，可调用环境文件类里面的自定义函数
    RL = DeepQNetwork(len(env.actions), 
                      len(env.scdict[1]),
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      output_graph=True
                      )
#    run_maze(env,RL,10000) # for train  the last parameter is # of iteration
    run_test(env,RL) # for test

#    RL.plot_cost()   

if __name__ == "__main__":
    main()

    


