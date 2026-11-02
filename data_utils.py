from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddObservedValuesIndicator,
    InstanceSplitter,
    ExpectedNumInstanceSampler,
    TestSplitSampler,
)
from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.dataset.repository import get_dataset

def create_data_loaders(dataset_name, context_length, prediction_length,
                        batch_size, num_batches_per_epoch):
    dataset = get_dataset(dataset_name)
    
    mask_unobserved = AddObservedValuesIndicator(
        target_field=FieldName.TARGET,
        output_field=FieldName.OBSERVED_VALUES,
    )
    training_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(
            num_instances=1,
            min_future=prediction_length,
        ),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )
    train_loader = TrainDataLoader(
        Cached(dataset.train),
        batch_size=batch_size,
        stack_fn=batchify,
        transform=mask_unobserved + training_splitter,
        num_batches_per_epoch=num_batches_per_epoch,
    )

    prediction_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=TestSplitSampler(),
        past_length=context_length,
        future_length=prediction_length,
        time_series_fields=[FieldName.OBSERVED_VALUES],
    )

    return dataset, mask_unobserved, prediction_splitter, train_loader