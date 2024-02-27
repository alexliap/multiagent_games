import matplotlib.pyplot as plt
import numpy as np

from game import GameAOne, GameATwo, GameBOne, GameBTwo


def test_1(epoch_range, game_type: str, loops: int, filename: str):
    kwargs = {
        "environment_size": (3, 3),
        "possible_tile_states": (0, 1, 2),
        "actions": (1, 2),
        "lr": 1e-4,
        "gamma": 0.96,
        "reward": 1,
        "neg_reward": 0,
    }
    mean_results = []
    std_results = []
    last_game_states = []
    x_epochs = []
    for epochs in epoch_range:
        results = []
        game_states = []
        for i in range(loops):
            if game_type == "A_1":
                game = GameAOne(**kwargs)
            elif game_type == "A_2":
                game = GameATwo(**kwargs)
            elif game_type == "B_1":
                game = GameBOne(**kwargs)
            elif game_type == "B_2":
                game = GameBTwo(**kwargs)
            game.play(epochs)
            game_states.append(game.last_game_state)
            results.append(game.exploitation_rewards[-1])

        case_2 = [1 for x in game_states if x == (1, 2, 1, 2, 1, 2, 1, 2, 1)]
        case_1 = [1 for x in game_states if x == (2, 1, 2, 1, 2, 1, 2, 1, 2)]

        mean_reward = np.mean(results).item()
        std_reward = np.std(results).item()

        mean_results.append(mean_reward)
        std_results.append(std_reward)
        last_game_states.append((len(case_1) + len(case_2)) / len(game_states))
        x_epochs.append(epochs)

        fig, (ax1, ax2) = plt.subplots(2, 1, layout="constrained", sharex=True)

        ax1.errorbar(x_epochs, mean_results, std_results, fmt="-o")
        ax1.set_ylabel("Cumulative Exploitation Reward")
        ax1.set_xlabel("Epochs")
        ax1.set_title(game_type)

        ax2.plot(x_epochs, last_game_states)
        ax2.set_ylabel("Success Rate")
        ax2.set_xlabel("Epochs")
        fig.savefig("experiments/test_1_" + game_type + "_" + filename)


def test_2(epoch_range, game_type: str, filename: str):
    kwargs = {
        "environment_size": (3, 3),
        "possible_tile_states": (0, 1, 2),
        "actions": (1, 2),
        "lr": 1e-4,
        "gamma": 0.95,
        "reward": 1,
        "neg_reward": 0,
    }
    results = []
    for epochs in epoch_range:
        if game_type == "A_1":
            game = GameAOne(**kwargs)
        elif game_type == "A_2":
            game = GameATwo(**kwargs)
        elif game_type == "B_1":
            game = GameBOne(**kwargs)
        elif game_type == "B_2":
            game = GameBTwo(**kwargs)
        game.play(epochs)
        results.append(game.exploitation_rewards[-1])
    # print(game.last_game_state)

    plt.scatter(epoch_range, results)
    plt.ylabel("Exploitation Reward")
    plt.xlabel("Epochs")
    plt.title(game_type)
    plt.savefig("experiments/test_2_" + game_type + "_" + filename)
    plt.close()


# game_types = ["B_1"]  # , "A_2", "B_1", "B_2"]
# for game_type in game_types:
#     test_1(
#         epoch_range=range(4000, 400000 + 1, 4000),
#         game_type=game_type,
#         loops=10,
#         filename="4000_400000_4000",
#     )

# for game_type in game_types:
#     test_2(
#         epoch_range=range(10_000, 200_000 + 1, 5_000),
#         game_type=game_type,
#         filename="10000_200000_5000",
#     )

# kwargs = {
#     "environment_size": (3, 3),
#     "possible_tile_states": (0, 1, 2),
#     "actions": (1, 2),
#     "lr": 1e-4,
#     "gamma": 0.96,
#     "reward": 1,
#     "neg_reward": -1,
# }
# game = GameBOne(**kwargs)
# game.play(60_001)
# a = game.agent_1.get_q_table()
# zero_pct = len(a[a==0.0])/(a.shape[0]*a.shape[1])
# b = game.exploitation_rewards[-1]
# print(zero_pct*100, b, a.shape[0], a.shape[1], game.last_game_state)
# diff = np.array(game.q_sums[1:]) - np.array(game.q_sums[:-1])
# diff = np.mean(diff.reshape(-1, 200), axis=1)
# plt.plot(range(diff.shape[0]), diff)
# plt.show()
# print(game.q_sums[-20:])
# print(a)
# import pickle
# with open(r"2x2_env_q_table.pkl", "wb") as output_file:
#     pickle.dump(a, output_file)
