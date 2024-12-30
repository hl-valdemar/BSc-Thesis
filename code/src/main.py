from models import ben_retry

if __name__ == "__main__":
    n_start = 3
    n_stop = 11

    step = 0
    while n_start + step <= n_stop:
        for _ in range(10):
            print(f"\nTraining BEN Model (chain length: {n_start + step})")
            ben_retry.main(chain_length=n_start + step, gamma=0.9)

        step += 2

    # step = 0
    # while n_start + step <= n_stop:
    #     for _ in tqdm(
    #         range(10), desc=f"Training GFlowNet Model (chain length: {n_start + step})"
    #     ):
    #         gflownet.train(n_start + step)
    #
    #     step += 2
