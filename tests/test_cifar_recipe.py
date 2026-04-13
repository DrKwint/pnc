from cifar_tasks import CIFARTrainMCDropoutPreActResNet18, CIFARTrainPreActResNet18
from training import _build_resnet_optimizer


def test_cifar_train_output_path_changes_with_optimizer():
    sgd_task = CIFARTrainPreActResNet18()
    adamw_task = CIFARTrainPreActResNet18(optimizer="adamw")

    assert sgd_task.output().path != adamw_task.output().path
    assert "_optsgd_" in sgd_task.output().path
    assert "_optadamw_" in adamw_task.output().path


def test_cifar_train_output_path_changes_with_augmentation():
    cutout_task = CIFARTrainPreActResNet18(cutout_size=8)
    no_cutout_task = CIFARTrainPreActResNet18(cutout_size=0)

    assert cutout_task.output().path != no_cutout_task.output().path
    assert "_augfcco8_" in cutout_task.output().path
    assert "_augfcco0_" in no_cutout_task.output().path


def test_mcdropout_training_path_includes_recipe_suffix():
    task = CIFARTrainMCDropoutPreActResNet18(optimizer="adamw", cutout_size=0)

    assert "_optadamw_" in task.output().path
    assert "_augfcco0_" in task.output().path


def test_optimizer_builder_supports_sgd_and_adamw():
    assert _build_resnet_optimizer("sgd", 0.1, 5e-4, 0.9, True) is not None
    assert _build_resnet_optimizer("adamw", 0.1, 5e-4, 0.9, True) is not None
