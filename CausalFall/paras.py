import argparse

def get_parameters():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument('--sisfall-dir', default=r"")
    parser.add_argument('--softfall-dir', default=r"")
    parser.add_argument('--kfall-dir', default=r"")
    parser.add_argument('--cgubes-dir', default=r"")
    parser.add_argument('--gcgraph-dir', default=r"")
    parser.add_argument('--save-path', default=r"")
    parser.add_argument('--save-fig', default=r"")

    parser.add_argument('--step-down', default=8, type=int)  #
    parser.add_argument('--sisfall-down', default=8, type=int)  # 200
    parser.add_argument('--kfall-down', default=4, type=int)  # 100
    parser.add_argument('--cgubes-down', default=8, type=int)  # 200

    parser.add_argument('--sensor-num', default=3, type=int, help='number of sensors')
    parser.add_argument('--pre-len', default=15, type=int, help='lead time')  # 15
    parser.add_argument('--front-len', default=71, type=int, help='before SMV')  # 75 27 38 49
    parser.add_argument('--rear-len', default=0, type=int, help='after SMV')  # 0 26 38 49
    parser.add_argument('--falling-len', default=72, type=int, help='falling')

    parser.add_argument('--modal-two', default=2, type=int)
    parser.add_argument('--modal-three', default=3, type=int)

    parser.add_argument('--d-pe', default=72, type=int)  # 76

    parser.add_argument('--d-model', default=72, type=int)  # 77 78
    parser.add_argument('--n-head', default=9, type=int)  # 11 13
    """
    Hz       downsample  total     n_head        pre    front
    20Hz        10/5      63          7           12      61      
    25Hz        8/4       77          7           15      75
    50Hz        4/2       145         5           30      143
    100Hz       2/1       297         9           60      295         
    200Hz       1         583         11          120     581         
    """

    parser.add_argument('--projected-dim', default=8, type=int)

    parser.add_argument('--predicted-dim', default=2, type=int)

    parser.add_argument('--epochs', default=500, type=int)
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.05, type=float)

    parser.add_argument('--dataset-name', default="kfall")  # sisfall slowfall kfall cgubes
    parser.add_argument('--axis-num', default=9, type=int, help='number of axis')
    parser.add_argument('--model-name', default='BiLSTM')
    parser.add_argument('--trial', default="Test")
    parser.add_argument('--isCE', default=1, type=int)  # 0 1
    parser.add_argument('--isPE', default=0, type=int)  # 0 1
    parser.add_argument('--isConv', default=2, type=int)  # 2 1
    parser.add_argument('--isDistill', default=1, type=int)  # 0 1
    parser.add_argument('--isHetero', default=0, type=int)  # 0 1

    parser.add_argument('--isProb', default=0, type=int)  # 0 1

    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    return parser.parse_args()
