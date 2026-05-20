python physicsnemo-curator/examples/structural_mechanics/crash/run_etl.py \
    --config-dir=fea-gtc-lab-2026/config \
    --config-name=crash_etl \
    serialization_format=zarr \
    etl.source.input_dir=/Users/phermosomore/Downloads/FEA/REPO/data/ \
    serialization_format.sink.output_dir=/Users/phermosomore/Downloads/FEA/REPO/processed_data/ \
    serialization_format.sink.compression_level=5 \
    serialization_format.sink.overwrite_existing=true \
    etl.processing.num_processes=8


python physicsnemo-curator/examples/structural_mechanics/crash/run_etl.py \
    --config-dir=fea-gtc-lab-2026/config \
    --config-name=crash_etl \
    serialization_format=vtp \
    etl.source.input_dir=/Users/phermosomore/Downloads/FEA/REPO/data/ \
    serialization_format.sink.output_dir=/Users/phermosomore/Downloads/FEA/REPO/processed_data/ \
    serialization_format.sink.overwrite_existing=true \
    etl.processing.num_processes=8