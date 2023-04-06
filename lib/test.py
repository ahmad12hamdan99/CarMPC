class Base:
    def __init__(self):
        self.okay = 1

    def plot(self):
        raise NotImplementedError


class Child(Base):
    def __init__(self):
        super().__init__()
        self.okay = 2

    def plot(self):
        print(f"Child class plot() : {self.okay}")


class GrandChild(Child):
    def __init__(self):
        super().__init__()
        self.okay = 3

    def plot(self):
        super().plot()
        print(f"GrandChild class plot() : {self.okay}")


if __name__ == '__main__':
    grandchild = GrandChild()
    grandchild.plot()
