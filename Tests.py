def main():
    dataset = open("Mushroom dataset/agaricus-lepiota.data", "r")
    line = (dataset.readline())
    while line != '':  # The EOF char is an empty string
        data = line.split(r",")
        print("edible" if data[0] == "e" else "poisonous", data[1:] )
        line = dataset.readline()


if __name__ == "__main__":
    main()
