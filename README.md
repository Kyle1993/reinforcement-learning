# Actor-Critic && DDPG
@(Reinforcement Learning)


##基本概念
####Stochastic Policy && Deterministic Policy
Stochastic Policy:：随机策略，输入state后，输出的是action的概率分布，无法处理连续的action
Deterministic Policy：固定策略，输入state后，输出的是一个确定的action值，可以处理连续的action
这两种策略被证明最后都可以收敛，如果是离散action，最好还是用Stochastic Policy，算法更稳定
####回合更新（MC） && 单步更新（TD）
回合更新：每次episode结束后更新，因为回合结束了，我们可以用$Q_t = R_t + \gamma R_{t-1}+\gamma^2R_{t-2}+ ... +\gamma^{t-1}R_1$得到整个episode中每个Q，作为$Q_{target}$
单步更新：在episode中，每走一步就更新一次，这种情况下我们只能根据$Q_t = R_t+Q_{t+1},其中Q_{t+1} = Critic(S_{t+1},A_{t+1})$来矫正

###训练tricks
* __对输入输入做normalization可以加速收敛，具体方法：使用gym自带的wrapper__
* __tan 和 lr 小一点，memory size 大点__
* __用一些方法 init weight 能加速训练，增加训练稳定性__
* __注意最末状态是没有next state的，计算Q的时候要特殊处理，具体方法：记录是否为Done，是的话×0__
* 注意查看 state和action的dim和value_space
* 记得state = next_state
* batch_norm 可能有负效果(可能而已)
* tanh 可能比 relu效果好，主要看具体任务的值域
* backward之后可以试试截断梯度
* test的时候可以把随机策略设置成只取概率最大的action

## Actor-Critic
[如何理解Actor-Critic-知乎](https://www.zhihu.com/question/56692640/answer/154994442)
Actor：输入state，输出action
Critic：输入State，Action，输出Q（离散空间下我们输入整个action空间的概率分布（一个向量，而非一个值）效果更好）
__更新Critic__：这里的$Q_{target}$有两种获取办法（回合更新和单步更新，计算公式如上述）
$$(Q_t = R_t + \gamma R_{t-1}+\gamma^2R_{t-2}+ ... +\gamma^{t-1}R_1) 或者 (Q_t = R_t+Q_{t+1},其中Q_{t+1} = Critic(S_{t+1},A_{t+1}))$$
$$Q_{eval}=Critic(S_t,A_t)$$
$$loss_{critic} = MSE_{loss}(Q_{target},Q_{eval})$$
__更新Actor__：这里根根据论文中证明的公式得到loss
$$TD_{error}=Q_{target}-Q_{eval}$$
$$loss_{actor} = -\log{prob_{a_t}} *TD_{error}$$
(TD_error可以表明这次action的选择的好坏程度，越好TD_error越大，如果TD_error<0说明这次的action选择还不如预期)
($ \log{prob_{a_t}}$是指这次所选择的action的概率取对数，有$Critic(S_t,A_t)$算出的概率分布，取对应的action index得到)
(我们的目标是为了最大化$ \log{prob_{a_t}} *TD_{error}$,所以loss取负)


## Advantage Actor Critic 
Advantate Actor Critic的基本思路和AC一样，只不过这里的Critic输入state输出V，且无法使用单步更新
因为计算Advantage需要Q值和V值，所以这里的Critic是输出V值的，Q值由回合更新方法得到。
__更新Critic__：
$$Q_t = R_t + \gamma R_{t-1}+\gamma^2R_{t-2}+ ... +\gamma^{t-1}R_1$$
$$V_{eval}=Critic(S_t)$$
$$V_{target}=R+\gamma *Critci(S_{t+1})$$
$$loss_{critic} = MSE_{loss}(V_{target},V_{eval})$$
__更新Actor__：
$$V_t=Critic(S_t)$$
$$Advantage_t = Q_t-V_t$$
$$loss_{actor} = -\log{prob_{a_t}} *Advantage_t$$
($ \log{prob_{a_t}}$是指这次所选择的action的概率取对数，有$Critic(S_t)$算出的概率分布，取对应的action index得到)
(我们的目标是为了最大化$ \log{prob_{a_t}} *Advantage_t$,所以loss取负)



##DDPG
[DDPG-知乎](https://zhuanlan.zhihu.com/p/26754280)
DDPG可以解决连续动作问题，应为他的actor只输出一个动作值，critic结合state和action，输出Q值。DDPG结合了actor-critic和DoubleDQN的思想，所以他有四个网络，actor, actor_targe, critic, critic_target。
![enter image description here](https://pic4.zhimg.com/v2-e35901a8f900e29d5c7a65cef9bb256b_b.png)
DDPG是用value-base的方式更新critic，用policy-gradient的方式更新actor。（这里面actor_targe和critic_target的更新是靠平滑拷贝,即每次BP的时候只有actor和critic被更新，他两更新完之后再把参数平滑拷贝到actor_target和critic_target)
__DDPG的训练能不能用回合更新还带验证（在gym上用回合更新崩了，可能需要调参）__

__更新Critic__：critic的最终目标是准确的打分（输入S和A，输出准确的Q值），所以 $loss_{critic} = (Q_{eval} - Q_{target})^2$，其中
$$A_{t+1} = Actor_{target}(S_{t+1})$$
$$Q_{t+1} = Critic_{target}(S_{t+1},A_{t+1})$$
$$Q_{target} = R + \gamma Q_{t+1}$$
$$Q_{eval}=Critic(S_t,A_t)$$
(这里的Q_{target}被现实的R修正过，我们认为它更准确，用来做target)

__更新Actor__：actor的最终目标是输出得分Q尽可能高的A，所以$loss_{actor}= - Q_t$（这里要最大化Q，所以把loss设置为-Q），其中：
$$A_t = Actor(S_t)$$
$$Q_t = Critic(S_t,A_t)$$

__几点问题__：
1. 在更新Critic里为什么用$Actor_{target}$计算$A_{t+1}$？
    因为我们希望这部分更新的是critic，所以使用actor_target(BP的时候它不会更新)
2. 在更新Critic里为什么用$Critic_{target}$计算$Q_{t+1}$？
	具体参见DoubleDQN，用target网络来计算$Q_t+1$可以解决__乐观估计的问题__
3. 在更新Actor里为什么用$Critic$来计算$Q_t$?
  这里Critic已经被更新过，且更新目标是被现实的R修正过的，我们认为它现在能更准确的估计Q
