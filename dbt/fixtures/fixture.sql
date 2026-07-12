create schema if not exists nyc_taxi_silver;

create or replace table nyc_taxi_silver.trips (
    trip_id varchar,
    service varchar,
    pu_ts timestamp,
    do_ts timestamp,
    pu_zone_id integer,
    do_zone_id integer,
    is_crz_pu boolean,
    is_buffer boolean,
    distance_mi double,
    duration_s bigint,
    avg_mph double,
    fare_ex_surcharge double,
    tip double,
    tip_rate double,
    cbd_congestion_fee double,
    source_file varchar,
    year_month varchar
);

insert into nyc_taxi_silver.trips values (
    'fixture-trip-1',
    'yellow',
    timestamp '2025-01-06 08:00:00',
    timestamp '2025-01-06 08:15:00',
    161,
    162,
    true,
    false,
    2.5,
    900,
    10.0,
    18.5,
    3.7,
    0.2,
    0.75,
    'fixture.parquet',
    '2025-01'
);

create or replace table nyc_taxi_silver.trips_quarantine (
    reason_code varchar,
    source_file varchar,
    service varchar,
    year_month varchar,
    raw_payload varchar
);

insert into nyc_taxi_silver.trips_quarantine values (
    'INVALID_FARE',
    'fixture-poisoned-parquet',
    'yellow',
    '2025-01',
    '{"fare_amount":-1}'
);
