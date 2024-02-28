import matplotlib.pyplot as plt
import numpy as np

from game import GameAOne, GameATwo, GameBOne, GameBTwo


def test_1(epoch_range, game_type: str, loops: int, args: dict, filename: str = ""):
    mean_results = []
    std_results = []
    last_game_states = []
    x_epochs = []
    for epochs in epoch_range:
        results = []
        game_states = []
        for i in range(loops):
            if game_type == "A_1":
                game = GameAOne(**args)
            elif game_type == "A_2":
                game = GameATwo(**args)
            elif game_type == "B_1":
                game = GameBOne(**args)
            elif game_type == "B_2":
                game = GameBTwo(**args)
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

        fig, (ax1, ax2) = plt.subplots(
            2, 1, layout="constrained", sharex=True, figsize=(6, 6)
        )

        ax1.errorbar(x_epochs, mean_results, std_results, fmt="-o")
        ax1.set_ylabel("Mean Cumulative Exploitation Reward")
        ax1.grid()
        ax1.set_title(game_type)

        ax2.plot(x_epochs, last_game_states)
        ax2.grid()
        ax2.axhline(1, linestyle="--", c="red")
        ax2.set_ylim([-0.2, 1.2])
        ax2.set_ylabel("Success Rate")
        ax2.set_xlabel("Epochs")
        fig.savefig("experiments/test_1_" + game_type + filename)


def test_2(epochs: int, game_type: str, args: dict, window: int, filename: str = ""):
    if game_type == "A_1":
        game = GameAOne(**args)
    elif game_type == "A_2":
        game = GameATwo(**args)
    elif game_type == "B_1":
        game = GameBOne(**args)
    elif game_type == "B_2":
        game = GameBTwo(**args)
    game.play(epochs)

    agent_cum_rewards_1 = np.array(game.rewards_1)
    agent_cum_rewards_2 = np.array(game.rewards_2)

    agent_cum_rewards_1 = np.cumsum(agent_cum_rewards_1, dtype=float)
    agent_cum_rewards_2 = np.cumsum(agent_cum_rewards_2, dtype=float)

    agent_cum_rewards_1[window:] = (
        agent_cum_rewards_1[window:] - agent_cum_rewards_1[:-window]
    )
    agent_cum_rewards_2[window:] = (
        agent_cum_rewards_2[window:] - agent_cum_rewards_2[:-window]
    )

    agent_cum_rewards_1[window - 1 :] = agent_cum_rewards_1[window - 1 :] / window
    agent_cum_rewards_2[window - 1 :] = agent_cum_rewards_2[window - 1 :] / window

    plt.plot(
        range(len(agent_cum_rewards_1[window - 1 :])), agent_cum_rewards_1[window - 1 :]
    )
    plt.plot(
        range(len(agent_cum_rewards_1[window - 1 :])), agent_cum_rewards_2[window - 1 :]
    )
    plt.xlim([0, epochs])
    plt.ylabel("Mean Cumulative Reward (Moving Average)")
    plt.xlabel("Epochs")
    plt.title(game_type)
    plt.legend(["Agent 1", "Agent 2"])
    plt.grid()

    plt.savefig("experiments/test_2_" + game_type + filename)
    plt.close()


kwargs = {
    "environment_size": (3, 3),
    "possible_tile_states": (0, 1, 2),
    "actions": (1, 2),
    "lr": 1e-1,
    "gamma": 0.96,
    "reward": 2.5,
}
game_types = ["B_2", "A_2"]
for game_type in game_types:
    test_1(
        epoch_range=range(500, 20000 + 1, 500),
        game_type=game_type,
        loops=20,
        args=kwargs,
    )

kwargs = {
    "environment_size": (3, 3),
    "possible_tile_states": (0, 1, 2),
    "actions": (1, 2),
    "lr": 1e-1,
    "gamma": 0.96,
    "reward": 1,
}
game_types = ["B_1", "A_1"]
for game_type in game_types:
    test_1(
        epoch_range=range(10_000, 150_000 + 1, 5_000),
        game_type=game_type,
        loops=10,
        args=kwargs,
    )

kwargs = {
    "environment_size": (3, 3),
    "possible_tile_states": (0, 1, 2),
    "actions": (1, 2),
    "lr": 1e-1,
    "gamma": 0.96,
    "reward": 1,
}
game_types = ["A_1", "B_1"]
for game_type in game_types:
    test_2(epochs=80000, game_type=game_type, args=kwargs, window=600)

kwargs = {
    "environment_size": (3, 3),
    "possible_tile_states": (0, 1, 2),
    "actions": (1, 2),
    "lr": 1e-1,
    "gamma": 0.96,
    "reward": 2.5,
}
game_types = ["B_2", "A_2"]
for game_type in game_types:
    test_2(epochs=15000, game_type=game_type, args=kwargs, window=500)
