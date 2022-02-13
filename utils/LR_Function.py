class LRPolicy():
    def __init__(self, epochs, decay_epoch, load_epoch):
        self.epochs = epochs
        self.decay_epoch = decay_epoch
        self.load_epoch = load_epoch

    def __call__(self, epoch):
        return 1.0 + max(0.0, epoch + self.load_epoch + 1. - self.decay_epoch)/(self.decay_epoch - self.epochs - 1.)