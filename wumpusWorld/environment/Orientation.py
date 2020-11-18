class North:
    def turnLeft(self):
        return West

    def turnRight(self):
        return East

class South:
    def turnLeft(self):
        return East

    def turnRight(self):
        return West

class East:
    def turnLeft(self):
        return North

    def turnRight(self):
        return South

class West:
    def turnLeft(self):
        return South

    def turnRight(self):
        return North