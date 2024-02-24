import matplotlib.pyplot as plt
import numpy as np

from game import GameAOne, GameATwo, GameBOne, GameBTwo


def test_1(epoch_range, game_type: str, loops: int, filename: str):
    kwargs = {
        "environment_size": (3, 3),
        "possible_tile_states": (0, 1, 2),
        "actions": (1, 2),
        "lr": 1e-4,
        "gamma": 0.95,
        "reward": 1,
        "neg_reward": 0,
    }
    mean_results = []
    std_results = []
    for epochs in epoch_range:
        results = []
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
            results.append(game.exploitation_rewards[-1])

        mean_reward = np.mean(results).item()
        std_reward = np.std(results).item()

        mean_results.append(mean_reward)
        std_results.append(std_reward)

    plt.errorbar(epoch_range, mean_results, std_results, fmt="-o")
    plt.ylabel("Mean Exploitation Reward")
    plt.xlabel("Epochs")
    plt.title(game_type)
    plt.savefig("experiments/test_1_" + game_type + "_" + filename)
    plt.close()


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

    plt.plot(epoch_range, results)
    plt.ylabel("Exploitation Reward")
    plt.xlabel("Epochs")
    plt.title(game_type)
    plt.savefig("experiments/test_2_" + game_type + "_" + filename)
    plt.close()


game_types = ["A_1", "A_2", "B_1", "B_2"]
for game_type in game_types:
    test_1(
        epoch_range=range(500, 10_000 + 1, 500),
        game_type=game_type,
        loops=10,
        filename="500_10000_500",
    )

for game_type in game_types:
    test_2(
        epoch_range=range(10_000, 200_000 + 1, 5_000),
        game_type=game_type,
        filename="10000_200000_5000",
    )

# t1 = np.arange(0, len(exploration))
# t2 = np.arange(len(exploration), epochs/100)

# plt.plot(t1, exploration, label = "Exploration")
# plt.plot(t2, exploitation, label = "Exploitation")
# plt.ylabel("Meam Cumalitve Reward")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()
