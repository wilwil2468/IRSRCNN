# wrap DataLoader to provide get_batch(batch_size) API
class BatchLoaderWrapper:
    def __init__(self, loader):
        self.loader = loader
        self.iter = iter(loader)
    def get_batch(self, batch_size):
        try:
            lr, hr = next(self.iter)
        except StopIteration:
            # restart epoch
            self.iter = iter(self.loader)
            lr, hr = next(self.iter)
        return lr, hr, None   # the 'None' is for the placeholder your model expects

# after you build train_loader and valid_loader:
train_set = BatchLoaderWrapper(train_loader)
valid_set = BatchLoaderWrapper(valid_loader)

# now call exactly as before:
srcnn.train(
    train_set,        # has get_batch
    valid_set,        # has get_batch
    steps=steps,
    batch_size=batch_size,   # still passed, though ignored by wrapper
    save_best_only=save_best_only,
    save_every=save_every,
    save_log=save_log,
    log_dir=ckpt_dir
)
