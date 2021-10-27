from FireModel import FireModel


def run_main():
    model = FireModel()
    model.generateRandVecField()
    model.computeSim(5)
    


if __name__ == "__main__":
    run_main()