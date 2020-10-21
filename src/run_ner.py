from src.data_proc import train_cv


def train_main():
    """kflods 折训练"""
    train_cv(ad_train=False, last_n_layer=4)
    # train_cv()


if __name__ == '__main__':
    train_main()
