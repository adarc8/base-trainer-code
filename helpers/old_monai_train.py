"""
This was a train function used in trainer to mimic original lightweightbraindiff train.
It was working good but my model reach pretty much same results so I ditched it.
"""
# from monai.utils import set_determinism
# from monai.engines import SupervisedTrainer
# from monai.handlers import MeanSquaredError, from_engine
# import ignite
#
# epochs_window = 10
# amount_of_10_epochs = n_epochs // epochs_window
# for _10_epochs in range(amount_of_10_epochs):
#     trainer = SupervisedTrainer(
#         device=self._device,
#         max_epochs=epochs_window,
#         train_data_loader=self._dataloaders[TRAIN],
#         network=self._model,
#         optimizer=self._opt,
#         loss_function=torch.nn.MSELoss(),
#         inferer=self._inferer,
#         prepare_batch=self._prepare_batch,
#         key_train_metric={
#             "train_acc": MeanSquaredError(reduction='mean', output_transform=from_engine(["pred", "label"]))},
#         amp=True
#     )
#     ignite.metrics.RunningAverage(output_transform=from_engine(["loss"], first=True)).attach(trainer, 'avg. loss')
#     from ignite.contrib.handlers import ProgressBar
#     ProgressBar().attach(trainer, ['avg. loss'])
#
#     # train the model
#     trainer.run()
#     self.save_output_to_disk(_10_epochs * 10)