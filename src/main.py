from FireModel import FireModel


def run_main():
    model = FireModel()
    model.generateRandVecField()
    print(model.v)


if __name__ == "__main__":
    run_main()