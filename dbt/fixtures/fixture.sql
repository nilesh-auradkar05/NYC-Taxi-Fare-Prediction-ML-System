create schema if not exists nyc_taxi_silver;

create or replace table nyc_taxi_silver.trips (
    trip_id varchar,
    service varchar,
    year_month varchar,
    source_file varchar,
    vendor_id varchar,
    passenger_count integer,
    pickup_ts timestamp,
    dropoff_ts timestamp,
    pu_zone_id integer,
    do_zone_id integer,
    pu_borough varchar,
    pu_zone varchar,
    pu_service_zone varchar,
    do_borough varchar,
    do_zone varchar,
    do_service_zone varchar,
    trip_distance decimal(12, 4),
    duration_s bigint,
    avg_mph double,
    fare_amount decimal(14, 4),
    extra decimal(14, 4),
    mta_tax decimal(14, 4),
    tip_amount decimal(14, 4),
    tolls_amount decimal(14, 4),
    improvement_surcharge decimal(14, 4),
    congestion_surcharge decimal(14, 4),
    airport_fee decimal(14, 4),
    cbd_congestion_fee decimal(14, 4),
    total_amount decimal(14, 4),
    fare_ex_surcharge decimal(14, 4),
    tip_rate double
);

insert into nyc_taxi_silver.trips values
    (
        'fixture-yellow-1', 'yellow', '2025-01', 'yellow-1.parquet', '1', 1,
        timestamp '2025-01-06 08:00:00', timestamp '2025-01-06 08:15:00',
        161, 162, 'Manhattan', 'Midtown Center', 'Yellow Zone',
        'Manhattan', 'Midtown East', 'Yellow Zone',
        2.5000, 900, 10.0, 18.5000, 1.0000, 0.5000, 3.7000, 0.0000,
        1.0000, 2.5000, 0.0000, 0.7500, 25.0000, 18.5000, 0.200000
    ),
    (
        'fixture-yellow-2', 'yellow', '2025-01', 'yellow-2.parquet', '1', 1,
        timestamp '2025-01-06 09:00:00', timestamp '2025-01-06 09:20:00',
        161, 163, 'Manhattan', 'Midtown Center', 'Yellow Zone',
        'Manhattan', 'Midtown North', 'Yellow Zone',
        4.0000, 1200, 12.0, 25.5000, 1.0000, 0.5000, 2.5500, 0.0000,
        1.0000, 2.5000, 0.0000, 0.7500, 35.0000, 25.5000, 0.100000
    ),
    (
        'fixture-hvfhv-1', 'hvfhv', '2025-01', 'hvfhv-1.parquet', 'HV0003', null,
        timestamp '2025-01-06 10:00:00', timestamp '2025-01-06 10:30:00',
        138, 161, 'Queens', 'LaGuardia Airport', 'Airports',
        'Manhattan', 'Midtown Center', 'Yellow Zone',
        10.0000, 1800, 20.0, 40.0000, 0.0000, 0.0000, 2.0000, 5.0000,
        0.0000, 2.7500, 0.0000, 1.5000, 50.0000, 40.0000, 0.050000
    );

create or replace table nyc_taxi_silver.trips_quarantine (
    quarantine_id varchar,
    trip_id varchar,
    reason_code varchar,
    source_file varchar,
    service varchar,
    year_month varchar,
    pickup_ts timestamp,
    dropoff_ts timestamp,
    pu_zone_id integer,
    do_zone_id integer,
    raw_record_json varchar
);

insert into nyc_taxi_silver.trips_quarantine values (
    'fixture-quarantine-1',
    'fixture-poisoned-1',
    'BAD_FARE',
    'fixture-poisoned.parquet',
    'yellow',
    '2025-01',
    timestamp '2025-01-06 11:00:00',
    timestamp '2025-01-06 11:10:00',
    161,
    162,
    '{"fare_amount":-1}'
);
