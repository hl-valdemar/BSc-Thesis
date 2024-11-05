import argparse
from lib.gridworld import GridWorld, Action
import q_learn

def setup_world(descriptor: str = "small", render: bool = False) -> GridWorld:
    description = ""
    if descriptor == "small":
        description = """
        ##########
        #s #     #
        #  #  #  #
        #  #  o  #
        #     # g#
        ##########
        """
    elif descriptor == "large":
        description = """
        ###################
        #s #     #g #     #
        #  #  #  #  #  #  #
        #  #  o  #  #  o  #
        #     #  #     #  #
        #oo####  #oo####  #
        #g #     #  #     #
        #  #  #     #  #  #
        #  #  o     #  o  #
        #     #  #     #g #
        ###################
        """
    else:
        raise ValueError(f"Unsupported world descriptor: {descriptor}")

    world = GridWorld(
        description,
        font_size=40,
        font_path="/usr/share/fonts/TTF/JetBrainsMonoNerdFontMono-Regular.ttf",
        render=render,
    )

    return world

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run reinforcement learning algorithms in a Grid World")
    parser.add_argument("-a", "--algorithm", choices=["qlearn"], help="The algorithm used for training", required=True)
    parser.add_argument("-s", "--world-size", choices=["small", "large"], default="small", help="The size of the world")
    parser.add_argument("-f", "--follow-agent", action="store_true", help="Follow the agent during training")
    parser.add_argument("-r", "--render", action="store_true", help="Render the GridWorld")
    parser.add_argument("-p", "--show-policy", action="store_true", help="Render the policy during training")
    parser.add_argument("-m", "--plot-metrics", action="store_true", help="Plot the training metrics")

    args = parser.parse_args()

    # Setup the world
    world = setup_world(args.world_size, args.render)

    # Run the specified algorithm
    if args.algorithm == "qlearn":
        Q, policy, metrics = world.run_training(
            lambda w: q_learn.trainer(
                w,
                num_episodes=50000,
                follow_agent=args.follow_agent,
                show_policy=args.show_policy
            ))

        print("Learned policy: (Q-learning)")
        for row in policy:
            print(' '.join([Action(action).name[0] for action in row]))

        if args.plot_metrics:
            q_learn.plot_metrics(metrics)
    else:
        raise ValueError(f"Unsupported algorithm: {args.algorithm}")

if __name__ == "__main__":
    main()
