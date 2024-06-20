def recover_old_array(new_array):
    n = len(new_array)
    old_array = [0] * n

    # 初始化第一个元素
    old_array[0] = new_array[0]

    # 用来累计前面所有old_array元素的和
    sum_old = old_array[0]

    for i in range(1, n):
        old_array[i] = new_array[i] - sum_old
        sum_old += old_array[i]  # 更新和

    return old_array


def calculate_discounted_rewards(rewards, gamma):
    """
    计算带折扣的累积奖励。
    :param rewards: 每轮的即时奖励数组。
    :param gamma: 折扣因子，介于0和1之间。
    :return: 每轮的累积奖励数组。
    """
    n = len(rewards)
    discounted_rewards = [0] * n  # 初始化累积奖励数组
    cumulative = 0  # 从最后一轮开始

    # 从后向前计算累积奖励
    for i in reversed(range(n)):
        cumulative = rewards[i] + gamma * cumulative
        discounted_rewards[i] = cumulative

    return discounted_rewards
def sum_with_for_loop(numbers):
    total = 0
    reward = []
    t=1
    for num in numbers:

        total += num
        reward.append(total)
        t+=1
    return reward

if __name__ == '__main__':
    # 示例
    new_array = [
1.201,
2.8945,
2.9803,
3.185,
4.4679,
-0.1878,
5.2327,
5.1037,
8.6986,
10.5374,
8.2092,
8.3496,
10.5594,
10.2063,
10.8391,
12.9574,
12.1073,
15.3218,
14.589,
18.0197,
16.7308,
16.9002,
16.3819,
18.6055,
18.5666,
17.8705,
20.0678,
19.2612,
18.1451,
21.2684,
19.9203,
21.2532,
20.1815,
21.8921,
21.2986,
21.7333,
21.4597,
21.9637,
23.505,
22.2784,
20.6305,
22.1348,
23.2646,
22.2006,
21.5686,
22.6675,
23.0803,
23.0636,
22.3978,
22.8745,
23.1031,
24.0413,
23.1748,
23.6145,
23.8178,
23.4729,
24.1715,
24.2725,
24.085,
24.0729,
23.5256,
24.2585,
24.1737,
23.9845,
24.7918,
23.5093,
23.7971,
23.308,
23.701,
23.0689,
23.2393,
23.4836,
23.2807,
23.2741,
24.1288,
24.4506,
24.0563,
24.2429,
23.9211,
23.6565,
24.9351,
23.8512,
24.2392,
23.581,
23.5924,
24.4233,
24.0678,
24.4625,
24.504,
24.0065,
24.0658,
24.1618,
25.0343,
24.4958,
24.665,
25.3496,
25.004,
24.0904,]
    old_array = recover_old_array(new_array)
    discounted_rewards=calculate_discounted_rewards(old_array, 0.9)
    #rewards=sum_with_for_loop(discounted_rewards)
    print("旧的数组是:", discounted_rewards)